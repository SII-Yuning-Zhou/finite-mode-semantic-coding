"""Profile-only edge codec paths for the frozen ICLR-MNIST carrier.

The functions in this module deliberately keep model loading, CDF-table
construction, and data I/O outside the timed regions.  A packet consists of a
32-bit little-endian payload-length prefix followed by one rANS stream; the
mode schedule is regenerated from the agreed seed at both terminals.
"""

from __future__ import annotations

import struct
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from .universal_offset_coder import UniversalOffsetEntropyCoder, stateless_level_indexes


@dataclass(frozen=True)
class EdgePacket:
    """In-memory packet; shape and schedule metadata are pre-shared/out-of-band."""

    framed_payload: bytes
    latent_shape: tuple[int, ...]
    levels: int
    seed: int
    conditioned: bool


def _cdf_args(coder: UniversalOffsetEntropyCoder, *, conditioned: bool, symbols: torch.Tensor, indexes: torch.Tensor) -> tuple[list[list[int]], list[int], list[int]]:
    cdf = coder._conditional_cdf if conditioned else coder._marginal_cdf
    cdf_lists = coder._conditional_cdf_lists if conditioned else coder._marginal_cdf_lists
    rows = int(cdf.shape[0])
    return cdf_lists, [int(cdf.shape[1])] * rows, [coder.symbol_minimum] * rows


@torch.no_grad()
def encode(codec: torch.nn.Module, coder: UniversalOffsetEntropyCoder, x_cpu: torch.Tensor, *, levels: int, seed: int, conditioned: bool, device: torch.device) -> EdgePacket:
    """Encode one CPU image, including input transfer, schedule, and rANS."""

    if codec.training or coder.entropy_bottleneck.training:
        raise RuntimeError("edge profiling requires a frozen eval-mode codec")
    x = x_cpu.to(device=device, non_blocking=False)
    latent = codec.encoder(x)
    indexes = stateless_level_indexes(latent.shape, levels=levels, seed=seed, device=device)
    symbols, _ = coder.quantize(latent, indexes)
    flat_indexes = coder._indexes(symbols, indexes, conditioned=conditioned).reshape(-1)
    cdf_lists, cdf_lengths, cdf_offsets = _cdf_args(coder, conditioned=conditioned, symbols=symbols, indexes=flat_indexes)
    from compressai import ans
    stream = ans.RansEncoder().encode_with_indexes(
        symbols.reshape(-1).long().cpu().tolist(),
        flat_indexes.long().cpu().tolist(),
        cdf_lists, cdf_lengths, cdf_offsets,
    )
    # The receiver already knows latent shape, levels, seed, and conditioning.
    framed = struct.pack("<I", len(stream)) + bytes(stream)
    return EdgePacket(framed, tuple(int(v) for v in latent.shape), levels, int(seed), conditioned)


@torch.no_grad()
def decode(codec: torch.nn.Module, coder: UniversalOffsetEntropyCoder, packet: EdgePacket, *, device: torch.device, return_details: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode one packet and return a CPU reconstruction (transfer included)."""

    if len(packet.framed_payload) < 4:
        raise ValueError("packet is shorter than the 32-bit frame header")
    size = struct.unpack("<I", packet.framed_payload[:4])[0]
    stream = packet.framed_payload[4:]
    if size != len(stream):
        raise ValueError("packet frame length does not match payload")
    shape = packet.latent_shape
    indexes = stateless_level_indexes(shape, levels=packet.levels, seed=packet.seed, device=device)
    # Decode indexes do not require the transmitted symbols.
    symbols_placeholder = torch.zeros(shape, dtype=torch.int32, device=device)
    flat_indexes = coder._indexes(symbols_placeholder, indexes, conditioned=packet.conditioned).reshape(-1)
    cdf_lists, cdf_lengths, cdf_offsets = _cdf_args(coder, conditioned=packet.conditioned, symbols=symbols_placeholder, indexes=flat_indexes)
    from compressai import ans
    decoded = ans.RansDecoder().decode_with_indexes(
        stream, flat_indexes.long().cpu().tolist(), cdf_lists, cdf_lengths, cdf_offsets
    )
    symbols = torch.tensor(decoded, dtype=torch.int32, device=device).reshape(shape)
    offsets = coder._offset_values.to(device)[indexes]
    latent = symbols.float() + coder._median_tensor(symbols) + offsets
    reconstruction = codec.postprocess_decoder_output(codec.decoder(latent)).cpu()
    return (reconstruction, latent, symbols) if return_details else reconstruction


@torch.no_grad()
def roundtrip_reference(codec: torch.nn.Module, coder: UniversalOffsetEntropyCoder, x_cpu: torch.Tensor, *, levels: int, seed: int, conditioned: bool, device: torch.device) -> dict[str, Any]:
    """Check packet and latent equivalence against the established audit path."""

    x = x_cpu.to(device=device)
    latent = codec.encoder(x)
    indexes = stateless_level_indexes(latent.shape, levels=levels, seed=seed, device=device)
    reference = coder.encode_decode(latent, indexes, conditioned=conditioned)
    packet = encode(codec, coder, x_cpu, levels=levels, seed=seed, conditioned=conditioned, device=device)
    if packet.framed_payload[4:] != reference.payload:
        raise RuntimeError("profile packet differs from encode_decode reference payload")
    reconstruction, profile_latent, profile_symbols = decode(codec, coder, packet, device=device, return_details=True)
    expected = codec.postprocess_decoder_output(codec.decoder(reference.decoded_latent)).cpu()
    if not torch.equal(profile_symbols.cpu(), reference.symbols.cpu()):
        raise RuntimeError("profile packet symbols differ from encode_decode reference")
    if not torch.equal(profile_latent.cpu(), reference.decoded_latent.cpu()):
        raise RuntimeError("profile packet latent differs from encode_decode reference")
    if not torch.equal(reconstruction, expected):
        raise RuntimeError("profile encode/decode reconstruction differs from encode_decode reference")
    return {"packet_bytes": len(packet.framed_payload), "payload_equal": True, "symbols_equal": True, "latent_equal": True, "reconstruction_equal": True}


class RSSSampler:
    """Sample process RSS during a short benchmark without perturbing the path."""

    def __init__(self, interval_seconds: float = 0.002) -> None:
        try:
            import psutil
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("edge profiling requires psutil") from exc
        self._process = psutil.Process()
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._peak = 0
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self._peak = max(self._peak, int(self._process.memory_info().rss))
            except (OSError, RuntimeError):
                pass
            self._stop.wait(self._interval)

    def __enter__(self) -> "RSSSampler":
        self._peak = int(self._process.memory_info().rss)
        self._baseline = self._peak
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._peak = max(self._peak, int(self._process.memory_info().rss))
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def baseline_bytes(self) -> int:
        return self._baseline

    @property
    def peak_bytes(self) -> int:
        return self._peak

    def set_baseline(self, value: int) -> None:
        self._baseline = int(value)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty timing sample")
    location = min(len(ordered) - 1, max(0.0, fraction * (len(ordered) - 1)))
    left, right = math.floor(location), math.ceil(location)
    weight = location - left
    return float(ordered[left] * (1.0 - weight) + ordered[right] * weight)


def summarize(values: list[float]) -> dict[str, Any]:
    return {"mean_ms": 1000.0 * sum(values) / len(values), "median_ms": 1000.0 * percentile(values, 0.5), "p95_ms": 1000.0 * percentile(values, 0.95), "raw_seconds": values}
