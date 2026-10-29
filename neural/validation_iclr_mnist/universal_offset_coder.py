"""Operational conditional entropy coding for universal-quantization offsets.

This module implements the diagnostic bridge that is absent from the original
ICLR-style MNIST carrier.  For a receiver-known offset ``U`` it forms

    J = round(Y - median - U),       Y_hat = J + median + U,

and codes exactly the same integer tensor either with offset-conditioned CDFs
``q(J|U)`` or with their uniform-offset marginal ``q(J)``.  Consequently, any
rate difference is caused only by whether the entropy coder knows ``U``; the
neural transform and reconstruction are identical.

The construction follows the operational convention used by TensorFlow
Compression's ``UniversalBatchedEntropyModel``: offset level ``l`` is
``(l + 1)/(L + 1) - 1/2``.  It deliberately remains a validation-only adapter
around a frozen CompressAI EntropyBottleneck.  It does not silently change the
training objective of an existing checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

try:
    from compressai import ans
    from compressai.entropy_models.entropy_models import pmf_to_quantized_cdf
except ImportError as exc:  # pragma: no cover - exercised by server preflight
    raise RuntimeError(
        "Universal offset coding requires CompressAI with the ANS extension."
    ) from exc


@dataclass(frozen=True)
class OffsetCodingResult:
    payload: bytes
    decoded_latent: torch.Tensor
    symbols: torch.Tensor
    level_indexes: torch.Tensor
    quantized_cdf_bits: float
    floating_model_bits: float
    out_of_support_symbols: int


def offset_values(levels: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the midpoint offset grid used by the published implementation."""

    if levels < 1:
        raise ValueError("levels must be positive.")
    indexes = torch.arange(levels, dtype=torch.float32, device=device)
    return (indexes + 1.0) / float(levels + 1) - 0.5


def stateless_level_indexes(
    shape: Sequence[int],
    *,
    levels: int,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Generate receiver-reproducible offset indexes without transmitting them."""

    if levels < 1:
        raise ValueError("levels must be positive.")
    # CPU generation gives one deterministic schedule across CUDA versions and
    # devices.  Only the agreed seed, tensor shape, and number of levels are
    # needed by the receiver.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
    indexes = torch.randint(
        low=0,
        high=levels,
        size=tuple(int(value) for value in shape),
        generator=generator,
        dtype=torch.long,
        device="cpu",
    )
    return indexes.to(device=device, non_blocking=False)


class UniversalOffsetEntropyCoder:
    """Build conditional and marginal ANS tables from one frozen prior.

    The conditional and marginal coders share the same finite symbol support,
    CDF precision, framing, symbols, and neural reconstruction.  The marginal
    PMF is defined as the uniform mixture of the conditional PMFs.  This makes
    the comparison an exact operational analogue of ``H(J)`` versus
    ``H(J|U)`` for the fitted prior.
    """

    def __init__(
        self,
        entropy_bottleneck,
        *,
        levels: int,
        symbol_minimum: int = -128,
        symbol_maximum: int = 128,
        precision: int = 16,
        probability_floor: float = 1e-12,
        escape_tail: bool = False,
    ) -> None:
        if levels < 1:
            raise ValueError("levels must be positive.")
        if symbol_minimum >= symbol_maximum:
            raise ValueError("symbol_minimum must be smaller than symbol_maximum.")
        if not 8 <= precision <= 20:
            raise ValueError("ANS precision must lie between 8 and 20 bits.")
        if escape_tail and precision != 16:
            raise ValueError("CompressAI's rANS escape/bypass backend requires 16-bit CDF precision.")
        if probability_floor <= 0:
            raise ValueError("probability_floor must be positive.")
        if entropy_bottleneck.training:
            raise RuntimeError("The entropy bottleneck must be frozen in eval mode.")

        self.entropy_bottleneck = entropy_bottleneck
        self.levels = int(levels)
        self.symbol_minimum = int(symbol_minimum)
        self.symbol_maximum = int(symbol_maximum)
        self.precision = int(precision)
        self.probability_floor = float(probability_floor)
        self.escape_tail = bool(escape_tail)

        quantiles = entropy_bottleneck.quantiles
        self.channels = int(quantiles.shape[0])
        self._device = quantiles.device
        self._medians = quantiles[:, 0, 1].detach().float()
        self._offset_values = offset_values(self.levels, device=self._device)

        (
            self._conditional_pmf,
            self._conditional_tail_mass,
            self._marginal_pmf,
            self._marginal_tail_mass,
            self.max_normalization_residual,
        ) = self._build_probability_tables()
        self.max_tail_mass = float(self._conditional_tail_mass.max())
        self._conditional_cdf = self._build_cdf_rows(
            self._conditional_pmf, self._conditional_tail_mass
        )
        self._marginal_cdf = self._build_cdf_rows(
            self._marginal_pmf, self._marginal_tail_mass
        )
        self._conditional_cdf_lists = self._conditional_cdf.tolist()
        self._marginal_cdf_lists = self._marginal_cdf.tolist()

    @torch.no_grad()
    def _build_probability_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        symbols = torch.arange(
            self.symbol_minimum,
            self.symbol_maximum + 1,
            dtype=torch.float32,
            device=self._device,
        )
        conditional_levels: list[torch.Tensor] = []
        conditional_tail_levels: list[torch.Tensor] = []
        normalization_residual = 0.0
        for level in range(self.levels):
            centers = (
                symbols.view(1, -1)
                + self._medians.view(self.channels, 1)
                + self._offset_values[level]
            )
            likelihood, lower, upper = self.entropy_bottleneck._likelihood(
                centers.reshape(self.channels, 1, -1), stop_gradient=True
            )
            pmf = likelihood[:, 0, :].float().clamp_min(self.probability_floor)
            # Match CompressAI's operational EntropyBottleneck update(): the
            # finite CDF receives one final escape symbol whose probability is
            # the fitted mass below/above the tabulated lattice.  The rANS
            # backend then transmits the signed overflow through its 4-bit
            # bypass path.  In particular, do not renormalize omitted tail mass
            # into ordinary symbols.
            tail = (
                torch.sigmoid(lower[:, 0, :1])
                + torch.sigmoid(-upper[:, 0, -1:])
            ).float().clamp_min(self.probability_floor)
            mass = pmf.sum(dim=-1, keepdim=True)
            fitted_mass = mass + tail if self.escape_tail else mass
            normalization_residual = max(
                normalization_residual,
                float((fitted_mass - 1.0).abs().max()),
            )
            conditional_levels.append(pmf if self.escape_tail else pmf / mass)
            conditional_tail_levels.append(
                tail if self.escape_tail else torch.zeros_like(tail)
            )
        conditional = torch.stack(conditional_levels, dim=1)
        conditional_tail = torch.stack(conditional_tail_levels, dim=1)
        marginal = conditional.mean(dim=1)
        marginal_tail = conditional_tail.mean(dim=1)
        return (
            conditional.cpu(),
            conditional_tail.cpu(),
            marginal.cpu(),
            marginal_tail.cpu(),
            normalization_residual,
        )

    def _build_cdf_rows(
        self, pmf: torch.Tensor, tail_mass: torch.Tensor
    ) -> torch.Tensor:
        flat = pmf.reshape(-1, pmf.shape[-1])
        tails = tail_mass.reshape(-1, 1)
        rows = [
            pmf_to_quantized_cdf(
                torch.cat((row, tail)) if self.escape_tail else row,
                self.precision,
            )
            for row, tail in zip(flat, tails)
        ]
        return torch.stack(rows).to(dtype=torch.int32, device="cpu")

    @property
    def support_size(self) -> int:
        return self.symbol_maximum - self.symbol_minimum + 1

    @property
    def conditional_table_bytes(self) -> int:
        rows = int(self._conditional_cdf.shape[0])
        # Include the int32 CDF, per-row CDF length, and symbol offset required
        # by indexed rANS decoding.
        return int(
            (self._conditional_cdf.numel() + 2 * rows)
            * self._conditional_cdf.element_size()
        )

    @property
    def marginal_table_bytes(self) -> int:
        rows = int(self._marginal_cdf.shape[0])
        return int(
            (self._marginal_cdf.numel() + 2 * rows)
            * self._marginal_cdf.element_size()
        )

    @property
    def shared_randomness_bits_per_coefficient(self) -> float:
        return math.log2(self.levels)

    def _median_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        return self._medians.to(latent.device).view(
            1, self.channels, *([1] * (latent.ndim - 2))
        )

    def quantize(
        self, latent: torch.Tensor, level_indexes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 4 or latent.shape[1] != self.channels:
            raise ValueError("Expected a BCHW latent with the fitted channel count.")
        if level_indexes.shape != latent.shape:
            raise ValueError("One receiver-known offset index is required per coefficient.")
        if level_indexes.dtype != torch.long:
            level_indexes = level_indexes.long()
        if torch.any(level_indexes < 0) or torch.any(level_indexes >= self.levels):
            raise ValueError("Offset level index lies outside the deployed table.")
        offsets = self._offset_values.to(latent.device)[level_indexes]
        median = self._median_tensor(latent)
        symbols = torch.round(latent.float() - median - offsets).to(torch.int32)
        reconstruction = symbols.float() + median + offsets
        return symbols, reconstruction

    def _indexes(self, symbols: torch.Tensor, level_indexes: torch.Tensor, *, conditioned: bool) -> torch.Tensor:
        channels = torch.arange(
            self.channels, device=symbols.device, dtype=torch.long
        ).view(1, self.channels, 1, 1)
        channels = channels.expand_as(symbols)
        if conditioned:
            return channels * self.levels + level_indexes
        return channels

    def _model_probabilities(
        self,
        symbols: torch.Tensor,
        level_indexes: torch.Tensor,
        *,
        conditioned: bool,
    ) -> torch.Tensor:
        symbol_positions = symbols.long().cpu() - self.symbol_minimum
        if not self.escape_tail:
            channel_indexes = torch.arange(self.channels, dtype=torch.long).view(
                1, self.channels, 1, 1
            ).expand_as(symbol_positions)
            if conditioned:
                levels = level_indexes.long().cpu()
                return self._conditional_pmf[
                    channel_indexes, levels, symbol_positions
                ]
            return self._marginal_pmf[channel_indexes, symbol_positions]
        in_support = (symbol_positions >= 0) & (symbol_positions < self.support_size)
        clipped_positions = symbol_positions.clamp(0, self.support_size - 1)
        channel_indexes = torch.arange(self.channels, dtype=torch.long).view(
            1, self.channels, 1, 1
        ).expand_as(symbol_positions)
        if conditioned:
            levels = level_indexes.long().cpu()
            ordinary = self._conditional_pmf[
                channel_indexes, levels, clipped_positions
            ]
            tail = self._conditional_tail_mass[channel_indexes, levels, 0]
            return torch.where(in_support, ordinary, tail)
        ordinary = self._marginal_pmf[channel_indexes, clipped_positions]
        tail = self._marginal_tail_mass[channel_indexes, 0]
        return torch.where(in_support, ordinary, tail)

    @staticmethod
    def _bypass_bits(symbols: torch.Tensor, minimum: int, maximum: int) -> float:
        """Return the exact raw-bit count used by CompressAI's rANS bypass."""

        values = symbols.long().cpu()
        raw = torch.where(
            values < minimum,
            -2 * (values - minimum) - 1,
            2 * (values - maximum - 1),
        )
        overflow = (values < minimum) | (values > maximum)
        raw = raw[overflow]
        if raw.numel() == 0:
            return 0.0
        n_bypass = torch.zeros_like(raw)
        remaining = raw.clone()
        while torch.any(remaining != 0):
            n_bypass += (remaining != 0).long()
            remaining >>= 4
        count_words = torch.div(n_bypass, 15, rounding_mode="floor") + 1
        return float((4 * (n_bypass + count_words)).sum())

    def encode_decode(
        self,
        latent: torch.Tensor,
        level_indexes: torch.Tensor,
        *,
        conditioned: bool,
    ) -> OffsetCodingResult:
        symbols, expected_latent = self.quantize(latent, level_indexes)
        out_of_support = int(
            ((symbols < self.symbol_minimum) | (symbols > self.symbol_maximum)).sum()
        )
        if out_of_support and not self.escape_tail:
            raise RuntimeError(
                f"{out_of_support} symbols fall outside "
                f"[{self.symbol_minimum}, {self.symbol_maximum}]."
            )
        indexes = self._indexes(
            symbols, level_indexes, conditioned=conditioned
        ).reshape(-1)
        symbols_flat = symbols.reshape(-1).long().cpu().tolist()
        indexes_flat = indexes.long().cpu().tolist()
        cdf = self._conditional_cdf if conditioned else self._marginal_cdf
        cdf_lists = (
            self._conditional_cdf_lists if conditioned else self._marginal_cdf_lists
        )
        cdf_lengths = [int(cdf.shape[1])] * int(cdf.shape[0])
        cdf_offsets = [self.symbol_minimum] * int(cdf.shape[0])

        encoder = ans.RansEncoder()
        payload = encoder.encode_with_indexes(
            symbols_flat,
            indexes_flat,
            cdf_lists,
            cdf_lengths,
            cdf_offsets,
        )
        decoder = ans.RansDecoder()
        decoded_flat = decoder.decode_with_indexes(
            payload,
            indexes_flat,
            cdf_lists,
            cdf_lengths,
            cdf_offsets,
        )
        decoded_symbols = torch.tensor(
            decoded_flat, dtype=torch.int32, device=latent.device
        ).reshape_as(symbols)
        if not torch.equal(decoded_symbols, symbols):
            raise RuntimeError("rANS did not reproduce the universal-quantization symbols.")
        offsets = self._offset_values.to(latent.device)[level_indexes]
        decoded_latent = decoded_symbols.float() + self._median_tensor(latent) + offsets
        if not torch.equal(decoded_latent, expected_latent):
            raise RuntimeError("Receiver offset reconstruction is not bit exact.")

        positions = symbols.long().cpu() - self.symbol_minimum
        if self.escape_tail:
            positions = torch.where(
                (positions >= 0) & (positions < self.support_size),
                positions,
                torch.full_like(positions, self.support_size),
            )
        cdf_cpu = cdf
        index_cpu = indexes.long().cpu().reshape_as(positions)
        low = cdf_cpu[index_cpu, positions]
        high = cdf_cpu[index_cpu, positions + 1]
        quantized_probabilities = (high - low).double() / float(1 << self.precision)
        if torch.any(quantized_probabilities <= 0):
            raise RuntimeError("Quantized CDF assigned zero probability to a symbol.")
        bypass_bits = (
            self._bypass_bits(symbols, self.symbol_minimum, self.symbol_maximum)
            if self.escape_tail
            else 0.0
        )
        quantized_bits = float((-torch.log2(quantized_probabilities)).sum()) + bypass_bits
        floating_probabilities = self._model_probabilities(
            symbols, level_indexes, conditioned=conditioned
        ).double()
        floating_bits = float((-torch.log2(floating_probabilities)).sum()) + bypass_bits
        return OffsetCodingResult(
            payload=payload,
            decoded_latent=decoded_latent,
            symbols=symbols,
            level_indexes=level_indexes,
            quantized_cdf_bits=quantized_bits,
            floating_model_bits=floating_bits,
            out_of_support_symbols=out_of_support,
        )
