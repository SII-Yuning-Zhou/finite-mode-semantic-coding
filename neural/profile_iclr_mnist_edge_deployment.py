"""Profile a frozen MNIST codec without training or receiver-head evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

from run_iclr_mnist_grid import expand_grid
from validation_iclr_mnist.config import _construct, canonical_fingerprint, protocol_fingerprint
from validation_iclr_mnist.data import load_mnist_splits
from validation_iclr_mnist.edge_profile import RSSSampler, decode, encode, roundtrip_reference, summarize, synchronize
from validation_iclr_mnist.model import ICLRMNISTCodec, model_resources
from validation_iclr_mnist.runtime import atomic_json, resolve_device, seed_everything
from validation_iclr_mnist.universal_offset_coder import UniversalOffsetEntropyCoder


PROFILE_PROTOCOL = "ICLR-MNIST-EDGE-PROFILE-V1"


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _select_entry(grid: Path, rate_weight: float, seed: int) -> dict[str, Any]:
    entries = [
        item for item in expand_grid(grid)
        if math.isclose(float(item["train"]["rate_weight"]), rate_weight, rel_tol=0.0, abs_tol=1e-12)
        and int(item["seed"]) == seed
    ]
    if len(entries) != 1:
        raise ValueError(f"Expected one grid entry for rate={rate_weight}, seed={seed}; found {len(entries)}")
    return entries[0]


def _state_bytes(module: torch.nn.Module) -> int:
    return sum(int(t.numel() * t.element_size()) for t in module.state_dict().values())


def _parameter_bytes(module: torch.nn.Module) -> int:
    return sum(int(t.numel() * t.element_size()) for t in module.parameters())


@torch.no_grad()
def _conv_macs(codec: ICLRMNISTCodec, sample: torch.Tensor) -> dict[str, Any]:
    totals = {"encoder": 0, "decoder": 0}
    handles = []
    def register(root: torch.nn.Module, name: str) -> None:
        def count(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if isinstance(module, torch.nn.Conv2d):
                value = output.numel() * math.prod(module.kernel_size) * module.in_channels // module.groups
            elif isinstance(module, torch.nn.ConvTranspose2d):
                value = inputs[0].numel() * math.prod(module.kernel_size) * module.out_channels // module.groups
            elif isinstance(module, torch.nn.Linear):
                value = output.numel() * module.in_features
            else:
                value = 0
            totals[name] += int(value)
        for module in root.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
                handles.append(module.register_forward_hook(count))
    register(codec.encoder, "encoder")
    register(codec.decoder, "decoder")
    try:
        codec.decoder(codec.encoder(sample))
    finally:
        for handle in handles:
            handle.remove()
    return {"encoder_conv_macs": totals["encoder"], "decoder_conv_macs": totals["decoder"], "codec_conv_macs": sum(totals.values()), "mac_accounting": "Conv2d/ConvTranspose2d/Linear multiply-accumulate counts, batch one; normalization, elementwise operations, and rANS reported through measured latency"}


def _memory_pass(operation: Callable[[int], object], *, device: torch.device, repetitions: int) -> dict[str, Any]:
    """Separate pass: sampled process RSS plus PyTorch CUDA allocator peaks."""

    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
    else:
        baseline_allocated = baseline_reserved = None
    with RSSSampler(interval_seconds=0.002) as sampler:
        for index in range(repetitions):
            operation(index)
        synchronize(device)
    result: dict[str, Any] = {
        "host_rss_baseline_bytes": sampler.baseline_bytes,
        "host_rss_peak_sampled_bytes": sampler.peak_bytes,
        "host_rss_peak_increment_sampled_bytes": max(0, sampler.peak_bytes - sampler.baseline_bytes),
        "host_rss_sampling_interval_ms": 2,
        "measurement": "separate warmed memory pass; process RSS sampled with psutil every 2 ms",
        "process_scope": "full frozen carrier (encoder, entropy prior, decoder) loaded; workload is the named sender or receiver path",
        "repetitions": repetitions,
    }
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        result.update({
            "cuda_allocated_baseline_bytes": baseline_allocated,
            "cuda_allocated_peak_bytes": peak_allocated,
            "cuda_allocated_peak_increment_bytes": max(0, peak_allocated - int(baseline_allocated)),
            "cuda_reserved_baseline_bytes": baseline_reserved,
            "cuda_reserved_peak_bytes": peak_reserved,
            "cuda_reserved_peak_increment_bytes": max(0, peak_reserved - int(baseline_reserved)),
        })
    return result


def _hardware(device: torch.device, threads: int) -> dict[str, Any]:
    import psutil
    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "system_memory_bytes": int(psutil.virtual_memory().total),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "requested_threads": threads,
        "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "precision": "float32",
        "amp_enabled": False,
        "tf32_enabled": False,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update({"cuda_device_name": properties.name, "cuda_total_memory_bytes": int(properties.total_memory), "cuda_compute_capability": [properties.major, properties.minor]})
    return result


@torch.inference_mode()
def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup < 0 or args.repetitions < 1 or args.samples < 1 or args.threads < 1:
        raise ValueError("warmup, repetitions, samples, and threads must be valid positive counts")
    if not args.levels or min(args.levels) < 1 or len(set(args.levels)) != len(args.levels):
        raise ValueError("levels must be distinct positive integers")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = resolve_device(args.device)
    entry = _select_entry(Path(args.grid), args.rate_weight, args.seed)
    config = _construct(entry)
    seed_everything(config.seed)
    checkpoint_path = Path(args.checkpoint or Path(config.output_dir) / "final.pt")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Frozen checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol_fingerprint") != protocol_fingerprint(config):
        raise RuntimeError("checkpoint/config protocol fingerprint mismatch")
    if config.stage != "s1_continuous" or config.dither.kind != "continuous":
        raise ValueError("profile requires the v7 continuous one-prior carrier")
    codec = ICLRMNISTCodec(
        base_channels=config.model.base_channels, latent_channels=config.model.latent_channels,
        dither_kind=config.dither.kind, mode_slots=len(config.dither.counts),
        codebook_seed=config.dither.codebook_seed,
        decoder_output_activation=config.model.decoder_output_activation,
        gdn_implementation=config.model.gdn_implementation,
        decoder_output_postprocess=config.model.decoder_output_postprocess,
    )
    codec.load_state_dict(checkpoint["codec"])
    del checkpoint
    codec.to(device=device, dtype=torch.float32).eval()
    codec.update_entropy_model(force=True)
    splits = load_mnist_splits(config.data.root, validation_size=config.data.validation_size, seed=config.data.split_seed, download=config.data.download, include_official_test=False)
    count = min(args.samples, len(splits.validation))
    records = [splits.validation[index] for index in range(count)]
    inputs = torch.stack([record["x"] for record in records]).float().cpu()
    validation_indices = [int(record["index"]) for record in records]
    del splits, records

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rate_label = f"{args.rate_weight:g}"
    output_path = output_dir / f"edge_profile_rate_{rate_label}_seed_{args.seed}_{device.type}.json"
    config_path = output_dir / f"edge_profile_rate_{rate_label}_seed_{args.seed}_{device.type}.config.json"
    atomic_json(config_path, entry)
    report: dict[str, Any] = {
        "schema_version": 1, "status": "RUNNING", "profile_protocol": PROFILE_PROTOCOL,
        "grid": str(Path(args.grid)), "rate_weight": config.train.rate_weight, "seed": config.seed,
        "config_fingerprint": canonical_fingerprint(entry), "protocol_fingerprint": protocol_fingerprint(config),
        "config_copy": str(config_path), "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path), "checkpoint_file_bytes": checkpoint_path.stat().st_size,
        "git_head": _git("rev-parse", "HEAD"), "git_status_porcelain": _git("status", "--porcelain"),
        "hardware": _hardware(device, args.threads),
        "measurement_protocol": {
            "batch_size": 1, "validation_input_count": count, "validation_indices": validation_indices,
            "timing_input_count": min(count, args.repetitions),
            "warmup_per_coder": args.warmup, "repetitions_per_coder": args.repetitions,
            "memory_pass_repetitions": args.repetitions, "paired_order": "conditional then marginal on even repetitions; reverse on odd repetitions",
            "timed_sender": "CPU input transfer + neural encoder + shared schedule + quantization + indexed rANS + 32-bit frame",
            "timed_receiver": "frame parse + regenerated schedule + indexed rANS decode + latent transfer + neural decoder + CPU output transfer",
            "excluded_setup": ["model/checkpoint loading", "CDF table construction", "dataset I/O"],
            "rss_observer_during_latency": False,
            "schedule_seed_base": config.dither.schedule_seed, "schedule_seed_rule": "base + 65537 * levels + repetition",
            "packet_frame_bits": 32, "shared_metadata": ["latent shape", "levels", "schedule seed", "conditioning selection"],
        },
        "model_resources": {
            **asdict(model_resources(codec)),
            "parameter_bytes": _parameter_bytes(codec), "model_state_bytes": _state_bytes(codec),
            "encoder_tensor_state_bytes": _state_bytes(codec.encoder), "decoder_tensor_state_bytes": _state_bytes(codec.decoder),
            "encoder_parameter_bytes": _parameter_bytes(codec.encoder), "decoder_parameter_bytes": _parameter_bytes(codec.decoder),
            "entropy_model_parameter_bytes": _parameter_bytes(codec.entropy_model),
            "entropy_tensor_state_bytes": _state_bytes(codec.entropy_model),
            "base_entropy_cdf_bytes": codec.entropy_model.cdf_table_bytes(),
            **_conv_macs(codec, inputs[:1].to(device)),
        }, "levels": [],
    }
    atomic_json(output_path, report)
    for levels in args.levels:
        coder = UniversalOffsetEntropyCoder(codec.entropy_model.models[0], levels=levels, symbol_minimum=-128, symbol_maximum=128, precision=16, escape_tail=True)
        checks = {}
        for conditioned in (True, False):
            name = "conditional" if conditioned else "marginal"
            checks[name] = roundtrip_reference(codec, coder, inputs[:1], levels=levels, seed=config.dither.schedule_seed + 65537 * levels, conditioned=conditioned, device=device)
        for index in range(args.warmup):
            x = inputs[index % count:index % count + 1]
            seed = config.dither.schedule_seed + 65537 * levels + index
            for conditioned in (True, False) if index % 2 == 0 else (False, True):
                packet = encode(codec, coder, x, levels=levels, seed=seed, conditioned=conditioned, device=device)
                decode(codec, coder, packet, device=device)
        synchronize(device)
        timings = {name: {"encode": [], "decode": [], "packet_bytes": []} for name in ("conditional", "marginal")}
        packets: dict[str, list[Any]] = {"conditional": [], "marginal": []}
        for index in range(args.repetitions):
            x = inputs[index % count:index % count + 1]
            seed = config.dither.schedule_seed + 65537 * levels + index
            for conditioned in (True, False) if index % 2 == 0 else (False, True):
                name = "conditional" if conditioned else "marginal"
                synchronize(device)
                started = time.perf_counter()
                packet = encode(codec, coder, x, levels=levels, seed=seed, conditioned=conditioned, device=device)
                synchronize(device)
                timings[name]["encode"].append(time.perf_counter() - started)
                started = time.perf_counter()
                decode(codec, coder, packet, device=device)
                synchronize(device)
                timings[name]["decode"].append(time.perf_counter() - started)
                timings[name]["packet_bytes"].append(len(packet.framed_payload))
                packets[name].append(packet)
        point: dict[str, Any] = {"levels": levels, "reference_checks": checks, "conditional_table_bytes": coder.conditional_table_bytes, "marginal_table_bytes": coder.marginal_table_bytes, "paths": {}}
        for conditioned in (True, False):
            name = "conditional" if conditioned else "marginal"
            def sender(index: int, conditioned: bool = conditioned) -> object:
                return encode(codec, coder, inputs[index % count:index % count + 1], levels=levels, seed=config.dither.schedule_seed + 65537 * levels + index, conditioned=conditioned, device=device)
            def receiver(index: int, name: str = name) -> object:
                return decode(codec, coder, packets[name][index % len(packets[name])], device=device)
            table_bytes = coder.conditional_table_bytes if conditioned else coder.marginal_table_bytes
            point["paths"][name] = {
                "encode": summarize(timings[name]["encode"]), "decode": summarize(timings[name]["decode"]),
                "packet_bytes": timings[name]["packet_bytes"],
                "static_sender_tensor_and_cdf_bytes": _state_bytes(codec.encoder) + _state_bytes(codec.entropy_model) + table_bytes,
                "static_receiver_tensor_and_cdf_bytes": _state_bytes(codec.decoder) + _state_bytes(codec.entropy_model) + table_bytes,
                "static_storage_accounting": "encoder or decoder state tensors + deployed entropy-prior state tensors + indexed-rANS CDF/length/offset int32 table",
                "sender_memory": _memory_pass(sender, device=device, repetitions=args.repetitions),
                "receiver_memory": _memory_pass(receiver, device=device, repetitions=args.repetitions),
            }
        report["levels"].append(point)
        atomic_json(output_path, report)
        print(f"[profile] rate={rate_label} device={device} L={levels} complete", flush=True)
    report["status"] = "PASS"
    atomic_json(output_path, report)
    print(f"[PASS] {output_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="configs/iclr_mnist/grid_s1_rate_calibration_v7.json")
    parser.add_argument("--rate-weight", required=True, type=float)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 3, 7, 15, 31])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output-dir", default="profile_output/iclr_mnist_edge_v1")
    args = parser.parse_args()
    run_profile(args)


if __name__ == "__main__":
    main()
