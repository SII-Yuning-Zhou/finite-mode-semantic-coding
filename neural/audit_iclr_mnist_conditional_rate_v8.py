"""Audit marginal versus shared-offset-conditioned operational rate.

The audit freezes an existing ICLR-aligned checkpoint.  For every requested
number of public offset levels it generates one reproducible shared schedule,
with an explicit repetition index for independent schedule repeats, and then
forms one latent reconstruction, and encodes its symbols twice:

* marginal control: one CDF per latent channel, q(J);
* conditioned coder: one CDF per channel and public offset level, q(J|U).

No neural parameters, reconstruction, labels, or samples differ between the
two payloads.  This isolates the operational value and deployment cost of
receiver-known common randomness.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from run_iclr_mnist_grid import expand_grid
from validation_iclr_mnist.config import (
    canonical_fingerprint,
    load_experiment_config,
    protocol_fingerprint,
)
from validation_iclr_mnist.data import load_mnist_splits
from validation_iclr_mnist.evaluation import _load_checkpoint, _loader
from validation_iclr_mnist.runtime import atomic_json, seed_everything
from validation_iclr_mnist.universal_offset_coder import (
    UniversalOffsetEntropyCoder,
    stateless_level_indexes,
)


AUDIT_PROTOCOL = "UQ-CONDITIONAL-RANS-V1"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _select_grid_entry(grid: Path, rate_weight: float) -> dict[str, object]:
    matches = [
        entry
        for entry in expand_grid(grid)
        if math.isclose(
            float(entry["train"]["rate_weight"]),
            float(rate_weight),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one grid entry for rate_weight={rate_weight}, found {len(matches)}."
        )
    return matches[0]


@torch.no_grad()
def _freeze_validation_latents(config, codec, device, maximum_samples: int):
    splits = load_mnist_splits(
        config.data.root,
        validation_size=config.data.validation_size,
        seed=config.data.split_seed,
        download=config.data.download,
        include_official_test=False,
    )
    loader = _loader(splits.validation, config)
    latents: list[torch.Tensor] = []
    degraded: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        if seen + len(x) > maximum_samples:
            keep = maximum_samples - seen
            x, hr, label = x[:keep], hr[:keep], label[:keep]
        if len(x) == 0:
            break
        latents.append(codec.encoder(x).cpu())
        degraded.append(x.cpu())
        targets.append(hr.cpu())
        labels.append(label.cpu())
        seen += len(x)
        if seen >= maximum_samples:
            break
    if seen < 1:
        raise RuntimeError("The validation audit received no samples.")
    return (
        torch.cat(latents),
        torch.cat(degraded),
        torch.cat(targets),
        torch.cat(labels),
    )


@torch.no_grad()
def run_audit(arguments: argparse.Namespace) -> dict[str, object]:
    grid_path = Path(arguments.grid)
    entry = _select_grid_entry(grid_path, arguments.rate_weight)
    output_root = Path(arguments.output)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    config_path = output_root.with_suffix(".config.json")
    config_path.write_text(
        json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    config, config_fingerprint, _ = load_experiment_config(config_path)
    checkpoint_path = Path(config.output_dir) / "final.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Frozen checkpoint is missing: {checkpoint_path}")

    device = torch.device(arguments.device)
    seed_everything(config.seed)
    codec, classifier, _, checkpoint = _load_checkpoint(
        config, checkpoint_path, device
    )
    if checkpoint.get("protocol_fingerprint") != protocol_fingerprint(config):
        raise RuntimeError("Checkpoint/config protocol fingerprint mismatch.")
    if config.stage != "s1_continuous" or len(codec.entropy_model.models) != 1:
        raise RuntimeError("The v8 conditional-rate audit requires the S1 one-prior carrier.")

    maximum_samples = min(arguments.maximum_samples, config.data.validation_size)
    latent_all, x_all, hr_all, labels_all = _freeze_validation_latents(
        config, codec, device, maximum_samples
    )
    latent_shape = list(latent_all.shape[1:])
    latent_coefficients_per_image = int(latent_all[0].numel())
    classifier.eval()
    entropy_bottleneck = codec.entropy_model.models[0]
    entropy_bottleneck.eval()

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "RUNNING",
        "scientific_claim_status": "PENDING",
        "audit_protocol": (
            "UQ-CONDITIONAL-RANS-ESCAPE-V2"
            if arguments.escape_tail
            else AUDIT_PROTOCOL
        ),
        "grid": str(grid_path),
        "grid_fingerprint": canonical_fingerprint(
            json.loads(grid_path.read_text(encoding="utf-8"))
        ),
        "config_fingerprint": config_fingerprint,
        "run_name": config.run_name,
        "rate_weight": config.train.rate_weight,
        "checkpoint": str(checkpoint_path),
        "checkpoint_protocol_fingerprint": checkpoint.get("protocol_fingerprint"),
        "split": "validation",
        "samples": int(len(latent_all)),
        "latent_shape": latent_shape,
        "latent_coefficients_per_image": latent_coefficients_per_image,
        "levels": list(arguments.levels),
        "schedule_repetition": int(arguments.schedule_repetition),
        "completed_levels": 0,
        "points": [],
    }
    atomic_json(output_root, report)

    for level_ordinal, levels in enumerate(arguments.levels, start=1):
        coder = UniversalOffsetEntropyCoder(
            entropy_bottleneck,
            levels=levels,
            symbol_minimum=arguments.symbol_minimum,
            symbol_maximum=arguments.symbol_maximum,
            precision=arguments.precision,
            escape_tail=arguments.escape_tail,
        )
        totals = {
            "conditional_actual_bits": 0.0,
            "marginal_actual_bits": 0.0,
            "conditional_quantized_cdf_bits": 0.0,
            "marginal_quantized_cdf_bits": 0.0,
            "conditional_floating_bits": 0.0,
            "marginal_floating_bits": 0.0,
            "conditional_codec_seconds": 0.0,
            "marginal_codec_seconds": 0.0,
            "mse_x_sum": 0.0,
            "mse_hr_sum": 0.0,
            "correct": 0,
            "pixels": 0,
            "samples": 0,
            "batches": 0,
        }
        symbol_minimum: int | None = None
        symbol_maximum: int | None = None
        bit_exact = True

        for batch_index, start in enumerate(
            range(0, len(latent_all), arguments.batch_size)
        ):
            stop = min(len(latent_all), start + arguments.batch_size)
            latent = latent_all[start:stop].to(device, non_blocking=True)
            x = x_all[start:stop].to(device, non_blocking=True)
            hr = hr_all[start:stop].to(device, non_blocking=True)
            labels = labels_all[start:stop].to(device, non_blocking=True)
            shared_seed = (
                int(config.dither.schedule_seed)
                + 1_000_003 * batch_index
                + 65_537 * int(levels)
                + 104_729 * int(arguments.schedule_repetition)
            )
            sender_levels = stateless_level_indexes(
                latent.shape,
                levels=levels,
                seed=shared_seed,
                device=device,
            )
            receiver_levels = stateless_level_indexes(
                latent.shape,
                levels=levels,
                seed=shared_seed,
                device=device,
            )
            if not torch.equal(sender_levels, receiver_levels):
                raise RuntimeError("Receiver did not regenerate the public offset schedule.")

            _synchronize(device)
            started = time.perf_counter()
            conditioned = coder.encode_decode(
                latent, sender_levels, conditioned=True
            )
            _synchronize(device)
            totals["conditional_codec_seconds"] += time.perf_counter() - started

            _synchronize(device)
            started = time.perf_counter()
            marginal = coder.encode_decode(
                latent, sender_levels, conditioned=False
            )
            _synchronize(device)
            totals["marginal_codec_seconds"] += time.perf_counter() - started

            if not torch.equal(conditioned.symbols, marginal.symbols):
                raise RuntimeError("Conditional and marginal coders changed the symbols.")
            if not torch.equal(conditioned.decoded_latent, marginal.decoded_latent):
                raise RuntimeError("Conditional and marginal coders changed the latent.")
            bit_exact = bit_exact and torch.equal(
                conditioned.decoded_latent,
                coder.quantize(latent, receiver_levels)[1],
            )
            raw_reconstruction = codec.decoder(conditioned.decoded_latent)
            reconstruction = codec.postprocess_decoder_output(raw_reconstruction)
            logits = classifier(reconstruction)

            # One uint32 payload-length prefix is counted for each independently
            # decodable validation block, identically for both coders.
            totals["conditional_actual_bits"] += 32 + 8 * len(conditioned.payload)
            totals["marginal_actual_bits"] += 32 + 8 * len(marginal.payload)
            totals["conditional_quantized_cdf_bits"] += conditioned.quantized_cdf_bits
            totals["marginal_quantized_cdf_bits"] += marginal.quantized_cdf_bits
            totals["conditional_floating_bits"] += conditioned.floating_model_bits
            totals["marginal_floating_bits"] += marginal.floating_model_bits
            totals["mse_x_sum"] += float(F.mse_loss(reconstruction, x, reduction="sum"))
            totals["mse_hr_sum"] += float(F.mse_loss(reconstruction, hr, reduction="sum"))
            totals["correct"] += int((logits.argmax(dim=1) == labels).sum())
            totals["pixels"] += int(reconstruction.numel())
            totals["samples"] += len(latent)
            totals["batches"] += 1
            batch_minimum = int(conditioned.symbols.min())
            batch_maximum = int(conditioned.symbols.max())
            symbol_minimum = (
                batch_minimum
                if symbol_minimum is None
                else min(symbol_minimum, batch_minimum)
            )
            symbol_maximum = (
                batch_maximum
                if symbol_maximum is None
                else max(symbol_maximum, batch_maximum)
            )

        pixels = int(totals["pixels"])
        samples = int(totals["samples"])
        framing_bits = 32.0 * int(totals["batches"])
        point = {
            "levels": levels,
            "status": "PASS" if bit_exact else "FAIL",
            "bitstream_roundtrip": "PASS_BIT_EXACT" if bit_exact else "FAIL",
            "same_symbols_and_reconstruction": True,
            "theoretical_iid_schedule_entropy_bits_per_coefficient": math.log2(levels),
            "theoretical_iid_schedule_entropy_bits_per_image": latent_coefficients_per_image
            * math.log2(levels),
            "engineering_prng_seed_bits_stored_per_stream": 64,
            "engineering_seed_bits_transmitted": 0,
            "public_schedule_assumption": "pre-shared seed and tensor shape",
            "conditional_table_rows": coder.channels * levels,
            "marginal_table_rows": coder.channels,
            "conditional_table_bytes": coder.conditional_table_bytes,
            "marginal_table_bytes": coder.marginal_table_bytes,
            "table_byte_ratio": coder.conditional_table_bytes
            / coder.marginal_table_bytes,
            "symbol_support": [coder.symbol_minimum, coder.symbol_maximum],
            "observed_symbol_range": [symbol_minimum, symbol_maximum],
            "max_support_normalization_residual": coder.max_normalization_residual,
            "max_escape_tail_mass": coder.max_tail_mass,
            "escape_tail_cdf_and_bypass": coder.escape_tail,
            "framing_bits": framing_bits,
            "conditional_actual_bpp": totals["conditional_actual_bits"] / pixels,
            "marginal_actual_bpp": totals["marginal_actual_bits"] / pixels,
            "conditional_quantized_cdf_bpp": (
                totals["conditional_quantized_cdf_bits"] + framing_bits
            )
            / pixels,
            "marginal_quantized_cdf_bpp": (
                totals["marginal_quantized_cdf_bits"] + framing_bits
            )
            / pixels,
            "conditional_floating_model_bpp": totals["conditional_floating_bits"]
            / pixels,
            "marginal_floating_model_bpp": totals["marginal_floating_bits"]
            / pixels,
            "actual_conditioning_gain_bpp": (
                totals["marginal_actual_bits"] - totals["conditional_actual_bits"]
            )
            / pixels,
            "quantized_cdf_conditioning_gain_bpp": (
                totals["marginal_quantized_cdf_bits"]
                - totals["conditional_quantized_cdf_bits"]
            )
            / pixels,
            "floating_model_conditioning_gain_bpp": (
                totals["marginal_floating_bits"] - totals["conditional_floating_bits"]
            )
            / pixels,
            "conditional_actual_minus_quantized_cdf_bpp": (
                totals["conditional_actual_bits"]
                - totals["conditional_quantized_cdf_bits"]
                - framing_bits
            )
            / pixels,
            "marginal_actual_minus_quantized_cdf_bpp": (
                totals["marginal_actual_bits"]
                - totals["marginal_quantized_cdf_bits"]
                - framing_bits
            )
            / pixels,
            "mse_x": totals["mse_x_sum"] / pixels,
            "mse_hr": totals["mse_hr_sum"] / pixels,
            "accuracy": totals["correct"] / samples,
            "conditional_codec_seconds": totals["conditional_codec_seconds"],
            "marginal_codec_seconds": totals["marginal_codec_seconds"],
            "conditional_images_per_second": samples
            / totals["conditional_codec_seconds"],
            "marginal_images_per_second": samples / totals["marginal_codec_seconds"],
            "samples": samples,
            "batches": int(totals["batches"]),
        }
        report["points"].append(point)
        report["completed_levels"] = level_ordinal
        atomic_json(output_root, report)

    level_one = next(point for point in report["points"] if point["levels"] == 1)
    level_one_gain = abs(float(level_one["actual_conditioning_gain_bpp"]))
    failures: list[str] = []
    if level_one_gain > 1e-12:
        failures.append("L=1 conditional and marginal payloads differ.")
    if any(point["bitstream_roundtrip"] != "PASS_BIT_EXACT" for point in report["points"]):
        failures.append("At least one rANS round trip is not bit exact.")
    if (not arguments.escape_tail) and any(point["observed_symbol_range"][0] < arguments.symbol_minimum for point in report["points"]):
        failures.append("Observed symbol fell below the declared support.")
    if (not arguments.escape_tail) and any(point["observed_symbol_range"][1] > arguments.symbol_maximum for point in report["points"]):
        failures.append("Observed symbol exceeded the declared support.")
    if any(
        point["max_support_normalization_residual"] > 1e-6
        for point in report["points"]
    ):
        failures.append(
            "The finite CDF support omitted more than 1e-6 fitted probability mass."
        )
    report["technical_gate"] = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checks": [
            "same frozen checkpoint and validation samples",
            "same symbols and neural reconstruction for both entropy coders",
            "stateless receiver offset regeneration",
            "bit-exact rANS round trip",
            "L=1 conditional/marginal identity control",
            (
                "out-of-support symbols use the same escape CDF and bit-exact bypass"
                if arguments.escape_tail
                else "observed symbols inside declared support"
            ),
            "finite-support body plus explicit tail mass residual at most 1e-6",
        ],
    }
    report["status"] = "PASS" if not failures else "FAIL"
    report["scientific_claim_status"] = (
        "READY_FOR_CROSS_CHECKPOINT_AGGREGATION" if not failures else "BLOCKED"
    )
    atomic_json(output_root, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--rate-weight", type=float, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 3, 7, 15, 31])
    parser.add_argument("--maximum-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--symbol-minimum", type=int, default=-128)
    parser.add_argument("--symbol-maximum", type=int, default=128)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--escape-tail", action="store_true")
    parser.add_argument("--schedule-repetition", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if 1 not in arguments.levels:
        raise ValueError("The correctness audit requires the L=1 identity control.")
    if len(set(arguments.levels)) != len(arguments.levels):
        raise ValueError("Offset levels must be unique.")
    if arguments.schedule_repetition < 0:
        raise ValueError("schedule-repetition must be nonnegative.")
    report = run_audit(arguments)
    print(json.dumps(report["technical_gate"], indent=2))


if __name__ == "__main__":
    main()
