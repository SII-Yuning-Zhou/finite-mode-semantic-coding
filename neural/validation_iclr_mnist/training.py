"""Training loop for the ICLR-aligned MNIST validation carrier."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import ExperimentConfig, protocol_fingerprint
from .data import MNISTSplits, load_mnist_splits
from .model import (
    Discriminator,
    GrayResNet18Small,
    ICLRMNISTCodec,
    gradient_penalty,
    model_resources,
)
from .runtime import (
    atomic_json,
    atomic_torch_save,
    dither_generator,
    draw_modes,
    seed_everything,
)


def build_models(
    config: ExperimentConfig, device: torch.device
) -> tuple[ICLRMNISTCodec, GrayResNet18Small, Discriminator]:
    codec = ICLRMNISTCodec(
        base_channels=config.model.base_channels,
        latent_channels=config.model.latent_channels,
        dither_kind=config.dither.kind,
        mode_slots=len(config.dither.counts),
        codebook_seed=config.dither.codebook_seed,
        decoder_output_activation=config.model.decoder_output_activation,
        gdn_implementation=config.model.gdn_implementation,
        decoder_output_postprocess=config.model.decoder_output_postprocess,
    ).to(device)
    classifier = GrayResNet18Small(width=config.model.classifier_width).to(device)
    discriminator = Discriminator(width=config.model.classifier_width).to(device)
    return codec, classifier, discriminator


def build_loaders(
    config: ExperimentConfig,
    *,
    include_official_test: bool = False,
) -> tuple[MNISTSplits, DataLoader, DataLoader, DataLoader]:
    splits = load_mnist_splits(
        config.data.root,
        validation_size=config.data.validation_size,
        seed=config.data.split_seed,
        download=config.data.download,
        include_official_test=include_official_test,
    )
    common = {
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    source_generator = torch.Generator().manual_seed(config.seed + 11)
    target_generator = torch.Generator().manual_seed(config.seed + 29)
    source_loader = DataLoader(
        splits.source_train,
        shuffle=True,
        drop_last=True,
        generator=source_generator,
        **common,
    )
    target_loader = DataLoader(
        splits.target_train,
        shuffle=True,
        drop_last=True,
        generator=target_generator,
        **common,
    )
    validation_loader = DataLoader(
        splits.validation, shuffle=False, drop_last=False, **common
    )
    return splits, source_loader, target_loader, validation_loader


def _cycle(loader: DataLoader):
    while True:
        yield from loader


def _unpair_target(
    source_indices: torch.Tensor,
    target_images: torch.Tensor,
    target_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically derange accidental equal-index positions in a batch."""

    if len(source_indices) <= 1:
        if torch.any(source_indices == target_indices):
            raise RuntimeError("Unpaired WGAN training needs batch_size > 1.")
        return target_images, target_indices
    for shift in range(len(source_indices)):
        if not torch.any(source_indices == target_indices):
            return target_images, target_indices
        target_images = torch.roll(target_images, shifts=1, dims=0)
        target_indices = torch.roll(target_indices, shifts=1, dims=0)
    raise RuntimeError("Could not construct an unpaired target batch.")


def pretrain_classifier(
    classifier: GrayResNet18Small,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> list[dict[str, float | int]]:
    epochs = config.train.classifier_pretrain_epochs
    if epochs == 0:
        return []
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=config.train.classifier_learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )

    def multiplier(epoch: int) -> float:
        warmup = min(config.train.classifier_warmup_epochs, epochs)
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    history: list[dict[str, float | int]] = []
    maximum_batches = config.train.smoke_max_steps
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        correct = 0
        samples = 0
        batches = 0
        for batch in loader:
            if maximum_batches is not None and batches >= maximum_batches:
                break
            images = batch["hr"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(images)
            correct += int((logits.argmax(1) == labels).sum())
            samples += len(images)
            batches += 1
        scheduler.step()
        if samples == 0:
            raise RuntimeError("Classifier pretraining produced no batches.")
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / samples,
                "accuracy": correct / samples,
                "samples": samples,
            }
        )
    return history


def train_experiment(
    config: ExperimentConfig,
    *,
    config_fingerprint: str,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(config.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.train.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.train.allow_tf32
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_metrics.jsonl"

    def archive_previous_attempt(path: Path) -> None:
        if not path.exists():
            return
        attempt = 1
        while True:
            archive = path.with_name(f"{path.stem}_attempt_{attempt:03d}{path.suffix}")
            if not archive.exists():
                path.replace(archive)
                return
            attempt += 1

    archive_previous_attempt(output_dir / "status.json")
    archive_previous_attempt(metrics_path)
    atomic_json(
        output_dir / "status.json",
        {"state": "RUNNING", "stage": "load_data", "run_name": config.run_name},
    )
    _, source_loader, target_loader, _ = build_loaders(config)
    codec, classifier, discriminator = build_models(config, device)
    if config.train.classifier_checkpoint is not None:
        classifier_path = Path(config.train.classifier_checkpoint)
        if not classifier_path.is_file():
            raise RuntimeError(
                f"Frozen classifier is missing: {classifier_path}. "
                "Run prepare_iclr_mnist_classifier.py first."
            )
        classifier_payload = torch.load(
            classifier_path, map_location=device, weights_only=False
        )
        classifier.load_state_dict(classifier_payload["classifier"])
        classifier_history = [
            {
                "loaded_frozen_classifier": str(classifier_path.resolve()),
                "classifier_seed": int(classifier_payload["seed"]),
            }
        ]
    else:
        classifier_history = pretrain_classifier(classifier, source_loader, config, device)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    classifier.eval()

    entropy_parameters = codec.entropy_main_parameters()
    entropy_aux_parameters = codec.entropy_aux_parameters()
    entropy_ids = {id(parameter) for parameter in codec.entropy_model.parameters()}
    transform_parameters = [
        parameter for parameter in codec.parameters() if id(parameter) not in entropy_ids
    ]
    primary_codec_parameters = transform_parameters + entropy_parameters
    generator_optimizer = torch.optim.Adam(
        [
            {"params": transform_parameters, "lr": config.train.learning_rate},
            {"params": entropy_parameters, "lr": config.train.entropy_learning_rate},
        ],
        betas=(0.5, 0.999),
    )
    entropy_aux_optimizer = torch.optim.Adam(
        entropy_aux_parameters,
        lr=config.train.entropy_aux_learning_rate,
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.train.discriminator_learning_rate,
        betas=(0.5, 0.999),
    )
    amp_enabled = config.train.mixed_precision and device.type == "cuda"
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.train.mixed_precision_dtype]
    if (
        amp_enabled
        and amp_dtype is torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "The frozen protocol requests CUDA bfloat16, but this GPU/runtime "
            "does not report bfloat16 support."
        )
    # BF16 has the FP32 exponent range and does not require loss scaling.
    # GradScaler remains active only for the legacy FP16 protocol.
    grad_scaler_enabled = amp_enabled and amp_dtype is torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=grad_scaler_enabled)
    source_iterator = _cycle(source_loader)
    target_iterator = _cycle(target_loader)
    total_steps = config.train.steps
    if config.train.smoke_max_steps is not None:
        total_steps = min(total_steps, config.train.smoke_max_steps)
    start = time.perf_counter()
    amp_skipped_steps = 0
    consecutive_amp_skips = 0

    def finite_scalar(name: str, value: torch.Tensor, step: int) -> None:
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(
                f"Non-finite {name} at training step {step}; "
                "the run is invalid and no final checkpoint may be emitted."
            )

    def finite_tensor(name: str, value: torch.Tensor, step: int) -> None:
        finite_mask = torch.isfinite(value)
        if bool(finite_mask.all()):
            return
        finite_count = int(finite_mask.sum().detach())
        total_count = value.numel()
        finite_values = value.detach()[finite_mask]
        finite_max_abs = (
            float(finite_values.float().abs().max())
            if finite_values.numel()
            else None
        )
        raise FloatingPointError(
            f"Non-finite {name} at training step {step}: "
            f"finite={finite_count}/{total_count}, dtype={value.dtype}, "
            f"finite_max_abs={finite_max_abs}; the run is invalid and no "
            "final checkpoint may be emitted."
        )

    for step in range(total_steps):
        source = next(source_iterator)
        target = next(target_iterator)
        x = source["x"].to(device, non_blocking=True)
        labels = source["label"].to(device, non_blocking=True)
        source_indices = source["index"].to(device, non_blocking=True)
        real = target["hr"].to(device, non_blocking=True)
        target_indices = target["index"].to(device, non_blocking=True)
        real, target_indices = _unpair_target(
            source_indices, real, target_indices
        )
        modes = draw_modes(config, len(x), offset=step, device=device)

        discriminator.train()
        critic_loss_value = 0.0
        gradient_penalty_value = 0.0
        for critic_index in range(config.train.critic_steps):
            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = codec(
                    x,
                    modes,
                    generator=dither_generator(
                        config,
                        offset=step * config.train.critic_steps + critic_index,
                        device=device,
                    ),
                )["reconstruction"]
            real_score = discriminator(real)
            fake_score = discriminator(fake)
            penalty = gradient_penalty(discriminator, real, fake)
            critic_loss = (
                fake_score.mean()
                - real_score.mean()
                + config.train.gradient_penalty * penalty
            )
            finite_scalar("critic_loss", critic_loss, step + 1)
            finite_scalar("gradient_penalty", penalty, step + 1)
            critic_loss.backward()
            discriminator_gradient_norm = nn.utils.clip_grad_norm_(
                discriminator.parameters(), config.train.gradient_clip
            )
            finite_scalar(
                "discriminator_gradient_norm", discriminator_gradient_norm, step + 1
            )
            discriminator_optimizer.step()
            critic_loss_value += float(critic_loss.detach())
            gradient_penalty_value += float(penalty.detach())

        codec.train()
        generator_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            output = codec(
                x,
                modes,
                generator=dither_generator(config, offset=9000000 + step, device=device),
            )
            reconstruction = output["reconstruction"]

        for name, value in (
            ("latent", output["latent"]),
            ("quantized_latent", output["quantized"]),
            ("reconstruction", reconstruction),
            ("rate_bits", output["rate_bits"]),
        ):
            finite_tensor(name, value, step + 1)

        # Losses and the two auxiliary receiver heads are precision islands.
        # Codec convolutions retain AMP acceleration, but bounded image losses,
        # entropy totals, WGAN scores, and task logits are accumulated in FP32.
        with torch.autocast(device_type=device.type, enabled=False):
            reconstruction_float = reconstruction.float()
            pixel_mse = F.mse_loss(reconstruction_float, x.float())
            if config.train.distortion_reduction == "per_sample_sum":
                # ICLR Eq. (B.1) uses E[||X-Y||_2^2], not pixel-mean MSE.
                # Keeping this conversion explicit prevents lambda_R from
                # changing meaning when image dimensions change.
                distortion = F.mse_loss(
                    reconstruction_float, x.float(), reduction="sum"
                ) / len(x)
            elif config.train.distortion_reduction == "uint8_pixel_mse":
                # ICLR Table 1 reports normalized MSE but its lambda sweep is
                # numerically compatible with the conventional 8-bit codec
                # objective.  This leaves reported MSE in [0,1] units while
                # balancing the training Lagrangian in intensity-squared units.
                distortion = pixel_mse * (255.0 ** 2)
            else:
                distortion = pixel_mse
            rate_bpp = output["rate_bits"].float().sum() / (len(x) * 28 * 28)
            adversarial = -discriminator(reconstruction_float).mean()
            classification = F.cross_entropy(
                classifier(reconstruction_float), labels
            )
            effective_rate_weight = config.train.effective_rate_weight
            loss = (
                distortion
                + effective_rate_weight * rate_bpp
                + config.train.adversarial_weight * adversarial
                + config.train.classification_weight * classification
            )
        for name, value in (
            ("distortion", distortion),
            ("rate_bpp", rate_bpp),
            ("adversarial", adversarial),
            ("classification", classification),
            ("loss", loss),
        ):
            finite_scalar(name, value, step + 1)
        scaler.scale(loss).backward()
        scaler.unscale_(generator_optimizer)
        codec_gradient_norm = nn.utils.clip_grad_norm_(
            primary_codec_parameters, config.train.gradient_clip
        )
        codec_gradient_finite = bool(torch.isfinite(codec_gradient_norm).all())
        if not grad_scaler_enabled and not codec_gradient_finite:
            finite_scalar("codec_gradient_norm", codec_gradient_norm, step + 1)
        amp_scale_before = float(scaler.get_scale())
        scaler.step(generator_optimizer)
        scaler.update()
        amp_scale_after = float(scaler.get_scale())
        amp_step_skipped = (
            grad_scaler_enabled and amp_scale_after < amp_scale_before
        )
        if grad_scaler_enabled and not codec_gradient_finite and not amp_step_skipped:
            raise FloatingPointError(
                f"Non-finite codec gradient at step {step + 1} was not handled "
                "by AMP GradScaler."
            )
        if amp_step_skipped:
            amp_skipped_steps += 1
            consecutive_amp_skips += 1
            if consecutive_amp_skips > 32:
                raise FloatingPointError(
                    f"AMP skipped more than 32 consecutive codec steps by step {step + 1}."
                )
        else:
            consecutive_amp_skips = 0

        # CompressAI's quantiles determine the finite CDF support used by the
        # operational ANS coder.  They are intentionally trained by their
        # dedicated auxiliary objective and optimizer, as in the official
        # CompressAI protocol.
        entropy_aux_optimizer.zero_grad(set_to_none=True)
        entropy_aux_loss = codec.entropy_aux_loss()
        finite_scalar("entropy_aux_loss", entropy_aux_loss, step + 1)
        entropy_aux_loss.backward()
        entropy_aux_gradient_norm = nn.utils.clip_grad_norm_(
            entropy_aux_parameters, config.train.gradient_clip
        )
        finite_scalar(
            "entropy_aux_gradient_norm", entropy_aux_gradient_norm, step + 1
        )
        entropy_aux_optimizer.step()

        record = {
            "step": step + 1,
            "loss": float(loss.detach()),
            "distortion_x": float(distortion.detach()),
            "distortion_pixel_mse_x": float(pixel_mse.detach()),
            "distortion_reduction": config.train.distortion_reduction,
            "proxy_bpp": float(rate_bpp.detach()),
            "raw_rate_weight": config.train.rate_weight,
            "rate_weight_scale": config.train.rate_weight_scale,
            "effective_rate_weight": effective_rate_weight,
            "adversarial": float(adversarial.detach()),
            "classification_ce": float(classification.detach()),
            "entropy_aux_loss": float(entropy_aux_loss.detach()),
            "entropy_aux_gradient_norm": float(entropy_aux_gradient_norm.detach()),
            "critic_loss": critic_loss_value / config.train.critic_steps,
            "gradient_penalty": gradient_penalty_value / config.train.critic_steps,
            "codec_gradient_norm": (
                float(codec_gradient_norm.detach()) if codec_gradient_finite else None
            ),
            "amp_scale_before": amp_scale_before,
            "amp_scale_after": amp_scale_after,
            "amp_step_skipped": amp_step_skipped,
            "amp_skipped_steps_cumulative": amp_skipped_steps,
            "compute_precision": (
                config.train.mixed_precision_dtype if amp_enabled else "float32"
            ),
            "tf32_enabled": config.train.allow_tf32,
            "grad_scaler_enabled": grad_scaler_enabled,
            "latent_max_abs": float(output["latent"].detach().float().abs().max()),
            "quantized_max_abs": float(
                output["quantized"].detach().float().abs().max()
            ),
            "reconstruction_min": float(reconstruction.detach().float().min()),
            "reconstruction_max": float(reconstruction.detach().float().max()),
            "elapsed_seconds": time.perf_counter() - start,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if (step + 1) % config.train.checkpoint_every == 0:
            atomic_torch_save(
                output_dir / f"checkpoint_step_{step + 1:06d}.pt",
                {
                    "schema_version": 1,
                    "step": step + 1,
                    "config_fingerprint": config_fingerprint,
                    "protocol_fingerprint": protocol_fingerprint(config),
                    "codec": codec.state_dict(),
                    "classifier": classifier.state_dict(),
                    "discriminator": discriminator.state_dict(),
                },
            )
        atomic_json(
            output_dir / "status.json",
            {
                "state": "RUNNING",
                "stage": "train_codec",
                "run_name": config.run_name,
                "step": step + 1,
                "total_steps": total_steps,
                "latest": record,
            },
        )

    for module_name, module in (
        ("codec", codec),
        ("discriminator", discriminator),
        ("classifier", classifier),
    ):
        for parameter_name, parameter in module.state_dict().items():
            if torch.is_tensor(parameter) and not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(
                    f"Non-finite {module_name}.{parameter_name} after training; "
                    "final checkpoint was not written."
                )

    checkpoint = output_dir / "final.pt"
    resources = asdict(model_resources(codec))
    atomic_torch_save(
        checkpoint,
        {
            "schema_version": 1,
            "step": total_steps,
            "config_fingerprint": config_fingerprint,
            "protocol_fingerprint": protocol_fingerprint(config),
            "config": asdict(config),
            "codec": codec.state_dict(),
            "classifier": classifier.state_dict(),
            "discriminator": discriminator.state_dict(),
            "classifier_pretraining": classifier_history,
            "resources": resources,
        },
    )
    summary = {
        "schema_version": 1,
        "status": "TRAINED",
        "run_name": config.run_name,
        "steps": total_steps,
        "checkpoint": str(checkpoint.resolve()),
        "config_fingerprint": config_fingerprint,
        "protocol_fingerprint": protocol_fingerprint(config),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "resources": resources,
        "classifier_pretraining": classifier_history,
        "compute_precision": (
            config.train.mixed_precision_dtype if amp_enabled else "float32"
        ),
        "tf32_enabled": config.train.allow_tf32,
        "grad_scaler_enabled": grad_scaler_enabled,
        "amp_skipped_steps": amp_skipped_steps,
        "raw_rate_weight": config.train.rate_weight,
        "rate_weight_scale": config.train.rate_weight_scale,
        "effective_rate_weight": config.train.effective_rate_weight,
        "elapsed_seconds": time.perf_counter() - start,
    }
    atomic_json(output_dir / "training_summary.json", summary)
    atomic_json(
        output_dir / "status.json",
        {"state": "TRAINED", "stage": "train_complete", **summary},
    )
    return summary
