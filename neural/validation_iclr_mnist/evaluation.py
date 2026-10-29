"""Operational bitstream, distribution, task, and M/O receiver evaluation."""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from validation_semcom.metrics import classification_metrics, multi_kernel_mmd_squared
from validation_semcom.schedule import DyadicSchedule

from .config import ExperimentConfig, protocol_fingerprint
from .data import load_mnist_splits
from .model import (
    COMPRESSAI_ENTROPY_PROTOCOL,
    COMPRESSAI_VERSION,
    EqualCapacityModeHead,
    GrayResNet18Small,
    ICLRMNISTCodec,
    model_resources,
)
from .runtime import atomic_json, dither_generator, draw_modes, seed_everything
from .training import build_models


def deployment_profile(
    codec: ICLRMNISTCodec,
    config: ExperimentConfig,
    device: torch.device,
    *,
    warmup: int = 10,
    repetitions: int = 50,
) -> dict[str, float | int | str | None]:
    """Count Conv/Linear MACs and measure batch-one end-to-end codec latency."""

    macs = 0
    encoder_macs = 0
    decoder_macs = 0
    hooks = []
    encoder_module_ids = {id(module) for module in codec.encoder.modules()}
    decoder_module_ids = {id(module) for module in codec.decoder.modules()}

    def hook(module, inputs, output):
        nonlocal macs, encoder_macs, decoder_macs
        increment = 0
        if isinstance(module, torch.nn.ConvTranspose2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            operations = kernel * module.out_channels // module.groups
            increment = int(inputs[0].numel() * operations)
        elif isinstance(module, torch.nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            operations = kernel * module.in_channels // module.groups
            increment = int(output.numel() * operations)
        elif isinstance(module, torch.nn.Linear):
            increment = int(output.numel() * module.in_features)
        macs += increment
        if id(module) in encoder_module_ids:
            encoder_macs += increment
        if id(module) in decoder_module_ids:
            decoder_macs += increment

    for module in codec.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    sample = torch.zeros(1, 1, 28, 28, device=device)
    mode = draw_modes(config, 1, offset=700000, device=device)
    with torch.no_grad():
        codec(
            sample,
            mode,
            generator=dither_generator(config, offset=700000, device=device),
        )
    for handle in hooks:
        handle.remove()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = 0
    with torch.no_grad():
        for index in range(warmup):
            codec(
                sample,
                mode,
                generator=dither_generator(
                    config, offset=710000 + index, device=device
                ),
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for index in range(repetitions):
            codec(
                sample,
                mode,
                generator=dither_generator(
                    config, offset=720000 + index, device=device
                ),
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encode_start = time.perf_counter()
        for index in range(repetitions):
            latent = codec.encoder(sample)
            dither_state = codec.quantizer.sample(
                latent,
                mode,
                generator=dither_generator(config, offset=730000 + index, device=device),
            )
            codec.entropy_model(
                latent.float() + dither_state["dither"].float(),
                dither_state["modes"],
                training=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encode_elapsed = time.perf_counter() - encode_start
    peak_increment = (
        max(0, torch.cuda.max_memory_allocated(device) - baseline_memory)
        if device.type == "cuda"
        else None
    )
    return {
        "batch_one_codec_macs": macs,
        "batch_one_encoder_macs": encoder_macs,
        "batch_one_decoder_macs": decoder_macs,
        "batch_one_codec_flops_2x_macs": 2 * macs,
        "batch_one_codec_latency_ms_mean": 1000.0 * elapsed / repetitions,
        "batch_one_encoder_entropy_proxy_latency_ms_mean": 1000.0
        * encode_elapsed
        / repetitions,
        "batch_one_peak_memory_increment_bytes": peak_increment,
        "latency_repetitions": repetitions,
        "device": str(device),
    }


def _loader(dataset, config: ExperimentConfig, *, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _load_checkpoint(
    config: ExperimentConfig, checkpoint_path: str | Path, device: torch.device
) -> tuple[ICLRMNISTCodec, GrayResNet18Small, torch.nn.Module, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    codec, classifier, discriminator = build_models(config, device)
    codec.load_state_dict(checkpoint["codec"])
    classifier.load_state_dict(checkpoint["classifier"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    codec.eval()
    codec.update_entropy_model(force=True)
    classifier.eval()
    discriminator.eval()
    return codec, classifier, discriminator, checkpoint


def sliced_wasserstein(
    x: np.ndarray, y: np.ndarray, *, projections: int, seed: int
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    count = min(len(x), len(y))
    if count < 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Sliced Wasserstein requires compatible nontrivial matrices.")
    generator = np.random.Generator(np.random.PCG64(seed))
    directions = generator.normal(size=(projections, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-12)
    x_projection = np.sort(x[:count] @ directions.T, axis=0)
    y_projection = np.sort(y[:count] @ directions.T, axis=0)
    return float(np.mean(np.abs(x_projection - y_projection)))


def _collect_receiver_training_data(
    codec: ICLRMNISTCodec,
    classifier: GrayResNet18Small,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    modes_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    seen = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if seen + len(x) > config.evaluation.calibration_samples:
                keep = config.evaluation.calibration_samples - seen
                x, labels = x[:keep], labels[:keep]
            if len(x) == 0:
                break
            modes = draw_modes(config, len(x), offset=300000 + batch_index, device=device)
            raw_reconstruction = codec(
                x,
                modes,
                generator=dither_generator(
                    config, offset=300000 + batch_index, device=device
                ),
            )["reconstruction"]
            reconstruction = codec.postprocess_decoder_output(raw_reconstruction)
            features.append(classifier.features(reconstruction).cpu())
            modes_all.append(modes.cpu())
            labels_all.append(labels.cpu())
            seen += len(x)
            if seen >= config.evaluation.calibration_samples:
                break
    return torch.cat(features), torch.cat(modes_all), torch.cat(labels_all)


def _train_receiver_pair(
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    validation_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, object]:
    train_features, train_modes, train_labels = train_data
    validation_features, validation_modes, validation_labels = validation_data
    heads = {
        "model_m": EqualCapacityModeHead(
            train_features.shape[1],
            mode_slots=len(config.dither.counts),
            mode_aware=True,
        ).to(device),
        "model_o": EqualCapacityModeHead(
            train_features.shape[1],
            mode_slots=len(config.dither.counts),
            mode_aware=False,
        ).to(device),
    }
    heads["model_o"].load_state_dict(heads["model_m"].state_dict())
    dataset = TensorDataset(train_features, train_modes, train_labels)
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed + 808),
    )
    reports: dict[str, object] = {}
    for name, head in heads.items():
        optimizer = torch.optim.Adam(
            head.parameters(), lr=config.evaluation.receiver_learning_rate
        )
        head.train()
        for _ in range(config.evaluation.receiver_epochs):
            for features, modes, labels in loader:
                features = features.to(device)
                modes = modes.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(features, modes), labels)
                loss.backward()
                optimizer.step()
        head.eval()
        with torch.no_grad():
            logits = head(
                validation_features.to(device), validation_modes.to(device)
            )
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        reports[name] = {
            "parameters": sum(parameter.numel() for parameter in head.parameters()),
            **classification_metrics(
                validation_labels.numpy(), probabilities, class_labels=tuple(range(10))
            ),
        }
    reports["paired_difference_m_minus_o"] = {
        "accuracy": reports["model_m"]["accuracy"] - reports["model_o"]["accuracy"],
        "cross_entropy": reports["model_m"]["cross_entropy"]
        - reports["model_o"]["cross_entropy"],
    }
    return reports


def evaluate_experiment(
    config: ExperimentConfig,
    *,
    config_fingerprint: str,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    evaluation_started = time.perf_counter()

    def update_status(stage: str, fraction: float, **details: object) -> None:
        atomic_json(
            output_dir / "status.json",
            {
                "state": "RUNNING",
                "stage": stage,
                "run_name": config.run_name,
                "evaluation_fraction": min(1.0, max(0.0, float(fraction))),
                "evaluation_elapsed_seconds": time.perf_counter() - evaluation_started,
                **details,
            },
        )

    update_status("evaluate_load", 0.0)
    if config.data.test_policy == "official_test":
        freeze_path = output_dir / "validation_freeze.json"
        if not freeze_path.is_file():
            raise RuntimeError(
                "Official MNIST test is locked. Create validation_freeze.json after "
                "freezing checkpoints, operating points, metrics, and figure rules."
            )
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("protocol_fingerprint") != protocol_fingerprint(config):
            raise RuntimeError("The official-test config differs from the frozen config.")
        if freeze.get("bitstream_protocol") != COMPRESSAI_ENTROPY_PROTOCOL:
            raise RuntimeError("The official-test bitstream protocol differs from validation.")
    splits = load_mnist_splits(
        config.data.root,
        validation_size=config.data.validation_size,
        seed=config.data.split_seed,
        download=config.data.download,
        include_official_test=config.data.test_policy == "official_test",
    )
    receiver_training_loader = _loader(splits.source_train, config)
    evaluation_dataset = (
        splits.test if config.data.test_policy == "official_test" else splits.validation
    )
    if evaluation_dataset is None:
        raise RuntimeError("Requested evaluation dataset is unavailable.")
    evaluation_loader = _loader(evaluation_dataset, config)
    codec, classifier, discriminator, checkpoint = _load_checkpoint(
        config, checkpoint_path, device
    )
    if checkpoint.get("protocol_fingerprint") != protocol_fingerprint(config):
        raise RuntimeError("Checkpoint/config fingerprint mismatch.")

    update_status("evaluate_entropy_model_ready", 0.03)
    total_actual_bits = 0
    total_proxy_bits = 0.0
    total_pixels = 0
    mse_x_sum = 0.0
    mse_hr_sum = 0.0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    features_reconstruction: list[np.ndarray] = []
    features_target: list[np.ndarray] = []
    receiver_features: list[torch.Tensor] = []
    receiver_modes: list[torch.Tensor] = []
    receiver_labels: list[torch.Tensor] = []
    critic_fake = 0.0
    critic_real = 0.0
    samples = 0
    evaluation_symbol_minimum: int | None = None
    evaluation_symbol_maximum: int | None = None
    raw_reconstruction_minimum: float | None = None
    raw_reconstruction_maximum: float | None = None
    raw_below_zero = 0
    raw_above_one = 0
    raw_values = 0
    encoded_streams = 0
    entropy_coding_batches = 0
    evaluation_sample_total = len(evaluation_dataset)

    update_status(
        "evaluate_bitstream",
        0.10,
        evaluation_samples=0,
        evaluation_sample_total=evaluation_sample_total,
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(evaluation_loader):
            x = batch["x"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if (
                config.evaluation.maximum_samples is not None
                and samples + len(x) > config.evaluation.maximum_samples
            ):
                keep = config.evaluation.maximum_samples - samples
                x, hr, labels = x[:keep], hr[:keep], labels[:keep]
            if len(x) == 0:
                break
            modes = draw_modes(config, len(x), offset=500000 + batch_index, device=device)
            output = codec.compress(
                x,
                modes,
                generator=dither_generator(
                    config, offset=500000 + batch_index, device=device
                ),
            )
            reconstruction = output["reconstruction"]
            raw_reconstruction = output["raw_reconstruction"]
            batch_raw_minimum = float(raw_reconstruction.min())
            batch_raw_maximum = float(raw_reconstruction.max())
            raw_reconstruction_minimum = (
                batch_raw_minimum
                if raw_reconstruction_minimum is None
                else min(raw_reconstruction_minimum, batch_raw_minimum)
            )
            raw_reconstruction_maximum = (
                batch_raw_maximum
                if raw_reconstruction_maximum is None
                else max(raw_reconstruction_maximum, batch_raw_maximum)
            )
            raw_below_zero += int((raw_reconstruction < 0).sum())
            raw_above_one += int((raw_reconstruction > 1).sum())
            raw_values += raw_reconstruction.numel()
            symbols = output["symbols"].cpu().numpy()
            strings = output["strings"]
            for sample_index in range(len(x)):
                values = [int(value) for value in symbols[sample_index].reshape(-1)]
                value_minimum = min(values)
                value_maximum = max(values)
                evaluation_symbol_minimum = (
                    value_minimum
                    if evaluation_symbol_minimum is None
                    else min(evaluation_symbol_minimum, value_minimum)
                )
                evaluation_symbol_maximum = (
                    value_maximum
                    if evaluation_symbol_maximum is None
                    else max(evaluation_symbol_maximum, value_maximum)
                )
            # Each mode block is serialized as uint32 byte length + payload.
            # Counting these prefixes makes the reported block payload
            # decodable without relying on Python list boundaries.
            total_actual_bits += sum(32 + 8 * len(payload) for payload in strings)
            encoded_streams += len(strings)
            entropy_coding_batches += 1
            decoded_reconstruction = output["decoded_reconstruction"]
            if not torch.allclose(
                decoded_reconstruction, reconstruction, atol=1e-6, rtol=1e-5
            ):
                raise RuntimeError("Decoded latent does not reproduce the reconstruction.")
            total_proxy_bits += float(output["rate_bits"].sum())
            total_pixels += len(x) * 28 * 28
            mse_x_sum += float(F.mse_loss(reconstruction, x, reduction="sum"))
            mse_hr_sum += float(F.mse_loss(reconstruction, hr, reduction="sum"))
            logits = classifier(reconstruction)
            probabilities_all.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            reconstructed_features = classifier.features(reconstruction)
            target_features = classifier.features(hr)
            receiver_features.append(reconstructed_features.cpu())
            receiver_modes.append(modes.cpu())
            receiver_labels.append(labels.cpu())
            remaining = config.evaluation.distribution_samples - sum(
                len(value) for value in features_reconstruction
            )
            if remaining > 0:
                features_reconstruction.append(reconstructed_features[:remaining].cpu().numpy())
                features_target.append(target_features[:remaining].cpu().numpy())
            critic_fake += float(discriminator(reconstruction).sum())
            critic_real += float(discriminator(hr).sum())
            samples += len(x)
            update_status(
                "evaluate_bitstream",
                0.10 + 0.45 * samples / max(1, evaluation_sample_total),
                evaluation_samples=samples,
                evaluation_sample_total=evaluation_sample_total,
                actual_bits_so_far=total_actual_bits,
            )
            if (
                config.evaluation.maximum_samples is not None
                and samples >= config.evaluation.maximum_samples
            ):
                break
    labels = np.concatenate(labels_all)
    probabilities = np.concatenate(probabilities_all)
    reconstructed = np.concatenate(features_reconstruction)
    target = np.concatenate(features_target)
    task = classification_metrics(labels, probabilities, class_labels=tuple(range(10)))
    update_status("evaluate_distribution", 0.60, evaluation_samples=samples)
    distribution = {
        "frozen_feature_mmd2": multi_kernel_mmd_squared(reconstructed, target),
        "frozen_feature_sliced_wasserstein": sliced_wasserstein(
            reconstructed,
            target,
            projections=config.evaluation.sliced_wasserstein_projections,
            seed=config.seed + 909,
        ),
        "wgan_critic_gap_real_minus_fake": (critic_real - critic_fake) / samples,
        "samples": len(reconstructed),
    }
    update_status("evaluate_receiver_features", 0.72, evaluation_samples=samples)
    receiver_train = _collect_receiver_training_data(
        codec, classifier, receiver_training_loader, config, device
    )
    receiver_validation = (
        torch.cat(receiver_features),
        torch.cat(receiver_modes),
        torch.cat(receiver_labels),
    )
    update_status("evaluate_receiver_heads", 0.84, evaluation_samples=samples)
    receiver = _train_receiver_pair(
        receiver_train, receiver_validation, config, device
    )
    if config.dither.kind == "finite":
        schedule = DyadicSchedule(config.dither.counts)
        common_randomness = {
            "K_slots": len(config.dither.counts),
            "K_active": sum(value > 0 for value in config.dither.counts),
            "counts": list(config.dither.counts),
            "H_V_bits_per_sample": schedule.entropy_bits,
            "exact_prefix_sampler_bits_per_sample": schedule.expected_exact_sampler_bits_per_symbol,
            "fixed_width_sampler_bits_per_sample": schedule.fixed_width_sampler_bits_per_symbol,
            "shared_schedule_payload_bits_per_sample": 0,
            "explicit_mode_signaling_bits_per_sample": math.ceil(
                math.log2(max(1, sum(value > 0 for value in config.dither.counts)))
            ),
        }
    else:
        common_randomness = {
            "type": "continuous_universal_dither_recovery_baseline",
            "finite_H_V_bits_per_sample": None,
            "shared_seed_payload_bits_per_sample": 0,
        }
    update_status("evaluate_deployment_profile", 0.95, evaluation_samples=samples)
    cdf_table_bytes = codec.entropy_model.cdf_table_bytes()
    resources = {
        **asdict(model_resources(codec)),
        **deployment_profile(codec, config, device),
        "dither_codebook_seed": config.dither.codebook_seed,
        "mode_schedule_seed": config.dither.schedule_seed,
        "stored_seed_bits_assuming_uint64_each": 128,
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "run_name": config.run_name,
        "split": config.data.test_policy,
        "samples": samples,
        "config_fingerprint": config_fingerprint,
        "protocol_fingerprint": protocol_fingerprint(config),
        "operating_point": {
            "stage": config.stage,
            "seed": config.seed,
            "rate_weight": config.train.rate_weight,
            "rate_weight_scale": config.train.rate_weight_scale,
            "effective_rate_weight": config.train.effective_rate_weight,
            "decoder_output_activation": config.model.decoder_output_activation,
            "gdn_implementation": config.model.gdn_implementation,
            "decoder_output_postprocess": config.model.decoder_output_postprocess,
            "dither_kind": config.dither.kind,
            "schedule_counts": list(config.dither.counts),
            "K_active": sum(value > 0 for value in config.dither.counts),
        },
        "rate_ledger": {
            "bitstream_protocol": COMPRESSAI_ENTROPY_PROTOCOL,
            "compressai_version": COMPRESSAI_VERSION,
            "entropy_model": "mode-conditioned EntropyBottleneck",
            "entropy_mode_models": codec.entropy_model.mode_slots,
            "entropy_coder": codec.entropy_model.entropy_coder_name,
            "entropy_coder_precision_bits": codec.entropy_model.entropy_coder_precision,
            "entropy_tail_mass": codec.entropy_model.tail_mass,
            "actual_latent_bpp": total_actual_bits / total_pixels,
            "entropy_model_proxy_bpp": total_proxy_bits / total_pixels,
            "actual_latent_bits": total_actual_bits,
            "pixels": total_pixels,
            "probability_conditioning": (
                "receiver-known finite mode V"
                if config.dither.kind == "finite"
                else "single entropy model for the continuous-dither baseline"
            ),
            "theory_rate_identity": (
                "H(Z|V), with H(Y|V) <= H(Z|V)"
                if config.dither.kind == "finite"
                else "operational latent cross-entropy for the continuous-dither baseline"
            ),
            "interpretation": "actual_latent_bpp is the operational block-ANS payload with one stream per active receiver-known mode and evaluation minibatch; finite-mode probability tables are selected by shared V, while mode/dither schedule metadata are reported separately",
            "entropy_cdf_table_bytes": cdf_table_bytes,
            "encoded_streams": encoded_streams,
            "entropy_coding_batches": entropy_coding_batches,
            "maximum_source_images_per_coding_batch": config.train.batch_size,
            "coding_block_policy": "one ANS stream per active receiver-known mode per evaluation minibatch",
            "stream_length_prefix_bits": 32,
            "stream_framing_bits_total": 32 * encoded_streams,
            "per_image_packet_framing_included": False,
            "evaluation_symbol_minimum": evaluation_symbol_minimum,
            "evaluation_symbol_maximum": evaluation_symbol_maximum,
        },
        "distortion": {
            "mse_to_degraded_input_x": mse_x_sum / total_pixels,
            "mse_to_paired_hr_reference": mse_hr_sum / total_pixels,
        },
        "decoder_range_diagnostics": {
            "raw_minimum": raw_reconstruction_minimum,
            "raw_maximum": raw_reconstruction_maximum,
            "raw_fraction_below_zero": raw_below_zero / max(1, raw_values),
            "raw_fraction_above_one": raw_above_one / max(1, raw_values),
            "operational_postprocess": config.model.decoder_output_postprocess,
        },
        "task": task,
        "target_distribution": distribution,
        "receiver_model_m_o": receiver,
        "common_randomness": common_randomness,
        "resources": resources,
        "bitstream_roundtrip": "PASS: CompressAI ANS payload reproduced the quantized latent and reconstruction",
    }
    report_path = output_dir / f"evaluation_{config.data.test_policy}.json"
    if report_path.is_file():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
        previous_rate_ledger = previous.get("rate_ledger")
        if (
            not isinstance(previous_rate_ledger, dict)
            or previous_rate_ledger.get("bitstream_protocol") != COMPRESSAI_ENTROPY_PROTOCOL
        ):
            archive = output_dir / f"evaluation_{config.data.test_policy}_pre_compressai_v0.json"
            if not archive.exists():
                shutil.copy2(report_path, archive)
    atomic_json(report_path, report)
    atomic_json(
        output_dir / "status.json",
        {"state": "COMPLETE", "stage": "evaluation_complete", **report},
    )
    return report


def freeze_validation_protocol(
    config: ExperimentConfig,
    *,
    config_fingerprint: str,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    """Lock the confirmatory test inputs after validation has been inspected."""

    if config.data.test_policy != "validation":
        raise ValueError("Freeze must be created from a validation configuration.")
    validation_report = Path(config.output_dir) / "evaluation_validation.json"
    if not validation_report.is_file():
        raise RuntimeError("Run validation evaluation before freezing the protocol.")
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    rate_ledger = report.get("rate_ledger")
    if (
        not isinstance(rate_ledger, dict)
        or rate_ledger.get("bitstream_protocol") != COMPRESSAI_ENTROPY_PROTOCOL
    ):
        raise RuntimeError("Validation must use the current CompressAI bitstream before freezing.")
    payload = {
        "schema_version": 1,
        "state": "FROZEN_BEFORE_OFFICIAL_TEST",
        "config_fingerprint": config_fingerprint,
        "protocol_fingerprint": protocol_fingerprint(config),
        "bitstream_protocol": COMPRESSAI_ENTROPY_PROTOCOL,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "validation_report": str(validation_report.resolve()),
        "frozen_items": [
            "checkpoint",
            "operating point",
            "metrics",
            "receiver protocol",
            "bitstream protocol",
            "plotting rules",
        ],
    }
    atomic_json(Path(config.output_dir) / "validation_freeze.json", payload)
    return payload
