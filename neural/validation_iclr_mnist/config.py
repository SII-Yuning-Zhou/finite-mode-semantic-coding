"""Validated configuration and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def canonical_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


@dataclass(frozen=True)
class DitherConfig:
    kind: str = "continuous"
    counts: tuple[int, ...] = (1,)
    codebook_seed: int = 1729
    schedule_seed: int = 2718

    def validate(self) -> None:
        if self.kind not in {"continuous", "finite"}:
            raise ValueError("dither.kind must be 'continuous' or 'finite'.")
        if not self.counts or any(value < 0 for value in self.counts):
            raise ValueError("dither.counts must be nonempty and nonnegative.")
        if sum(value > 0 for value in self.counts) == 0:
            raise ValueError("At least one dither mode must be active.")
        total = sum(self.counts)
        if total & (total - 1):
            raise ValueError("dither.counts must sum to a power of two.")
        if self.kind == "continuous" and self.counts != (1,):
            raise ValueError("Continuous dither does not use a finite mode schedule.")


@dataclass(frozen=True)
class ModelConfig:
    base_channels: int = 48
    latent_channels: int = 12
    classifier_width: int = 32
    decoder_output_activation: str = "sigmoid"
    gdn_implementation: str = "legacy_softplus"
    decoder_output_postprocess: str = "identity"

    def validate(self) -> None:
        if min(self.base_channels, self.latent_channels, self.classifier_width) < 1:
            raise ValueError("All model widths must be positive.")
        if self.decoder_output_activation not in {"sigmoid", "identity"}:
            raise ValueError(
                "model.decoder_output_activation must be 'sigmoid' or 'identity'."
            )
        if self.gdn_implementation not in {"legacy_softplus", "compressai"}:
            raise ValueError(
                "model.gdn_implementation must be 'legacy_softplus' or 'compressai'."
            )
        if self.decoder_output_postprocess not in {"identity", "clamp_0_1"}:
            raise ValueError(
                "model.decoder_output_postprocess must be 'identity' or 'clamp_0_1'."
            )


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    batch_size: int = 512
    num_workers: int = 4
    learning_rate: float = 5e-3
    entropy_learning_rate: float = 1e-4
    entropy_aux_learning_rate: float = 1e-3
    discriminator_learning_rate: float = 1e-4
    classifier_learning_rate: float = 5e-2
    classifier_pretrain_epochs: int = 10
    classifier_warmup_epochs: int = 2
    classifier_checkpoint: str | None = None
    rate_weight: float = 1000.0
    rate_weight_scale: float = 1.0
    adversarial_weight: float = 0.02
    classification_weight: float = 0.01
    distortion_reduction: str = "pixel_mean"
    gradient_penalty: float = 10.0
    critic_steps: int = 5
    gradient_clip: float = 2.0
    mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"
    allow_tf32: bool = False
    checkpoint_every: int = 250
    smoke_max_steps: int | None = None

    @property
    def effective_rate_weight(self) -> float:
        return self.rate_weight * self.rate_weight_scale

    def validate(self) -> None:
        positive = (
            self.steps,
            self.batch_size,
            self.learning_rate,
            self.entropy_learning_rate,
            self.entropy_aux_learning_rate,
            self.discriminator_learning_rate,
            self.classifier_learning_rate,
            self.rate_weight_scale,
            self.critic_steps,
            self.gradient_clip,
            self.checkpoint_every,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Training counts, rates, and clipping must be positive.")
        if (
            self.num_workers < 0
            or self.classifier_pretrain_epochs < 0
            or self.classifier_warmup_epochs < 0
        ):
            raise ValueError("Worker and pretraining counts cannot be negative.")
        if (
            self.classifier_pretrain_epochs > 0
            and self.classifier_warmup_epochs > self.classifier_pretrain_epochs
        ):
            raise ValueError("Classifier warm-up cannot exceed pretraining epochs.")
        if min(
            self.rate_weight,
            self.adversarial_weight,
            self.classification_weight,
            self.gradient_penalty,
        ) < 0:
            raise ValueError("Loss weights cannot be negative.")
        if self.mixed_precision_dtype not in {"float16", "bfloat16"}:
            raise ValueError(
                "train.mixed_precision_dtype must be 'float16' or 'bfloat16'."
            )
        if self.distortion_reduction not in {
            "pixel_mean",
            "per_sample_sum",
            "uint8_pixel_mse",
        }:
            raise ValueError(
                "train.distortion_reduction must be 'pixel_mean', "
                "'per_sample_sum', or 'uint8_pixel_mse'."
            )


@dataclass(frozen=True)
class DataConfig:
    root: str = "data_external/mnist"
    validation_size: int = 5000
    split_seed: int = 20260801
    download: bool = False
    test_policy: str = "validation"

    def validate(self) -> None:
        if self.validation_size < 1 or self.validation_size >= 60000:
            raise ValueError("validation_size must lie between 1 and 59999.")
        if self.split_seed < 0:
            raise ValueError("data.split_seed must be nonnegative.")
        if self.test_policy not in {"validation", "official_test"}:
            raise ValueError("test_policy must be validation or official_test.")


@dataclass(frozen=True)
class EvaluationConfig:
    calibration_samples: int = 10000
    maximum_samples: int | None = None
    symbol_minimum: int = -64
    symbol_maximum: int = 64
    distribution_samples: int = 2000
    sliced_wasserstein_projections: int = 128
    receiver_epochs: int = 20
    receiver_learning_rate: float = 1e-2

    def validate(self) -> None:
        if self.calibration_samples < 1 or self.distribution_samples < 2:
            raise ValueError("Calibration and distribution sample counts are too small.")
        if self.maximum_samples is not None and self.maximum_samples < 1:
            raise ValueError("maximum_samples must be positive when specified.")
        if self.symbol_minimum >= self.symbol_maximum:
            raise ValueError("symbol_minimum must be smaller than symbol_maximum.")
        if self.sliced_wasserstein_projections < 1 or self.receiver_epochs < 0:
            raise ValueError("Projection count must be positive and epochs nonnegative.")
        if self.receiver_learning_rate <= 0:
            raise ValueError("receiver_learning_rate must be positive.")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    run_name: str
    seed: int
    stage: str
    output_dir: str
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dither: DitherConfig = field(default_factory=DitherConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Only experiment schema_version=1 is supported.")
        if not self.run_name or not self.output_dir:
            raise ValueError("run_name and output_dir are required.")
        if self.stage not in {"s1_continuous", "s2_finite"}:
            raise ValueError("stage must be s1_continuous or s2_finite.")
        self.data.validate()
        self.model.validate()
        self.dither.validate()
        self.train.validate()
        self.evaluation.validate()
        if self.stage == "s1_continuous" and self.dither.kind != "continuous":
            raise ValueError("S1 must use continuous dither.")
        if self.stage == "s2_finite" and self.dither.kind != "finite":
            raise ValueError("S2 must use finite dither.")


def _construct(payload: dict[str, Any]) -> ExperimentConfig:
    config = ExperimentConfig(
        schema_version=int(payload["schema_version"]),
        run_name=str(payload["run_name"]),
        seed=int(payload["seed"]),
        stage=str(payload["stage"]),
        output_dir=str(payload["output_dir"]),
        data=DataConfig(**payload.get("data", {})),
        model=ModelConfig(**payload.get("model", {})),
        dither=DitherConfig(
            **{
                **payload.get("dither", {}),
                "counts": tuple(payload.get("dither", {}).get("counts", (1,))),
            }
        ),
        train=TrainConfig(**payload.get("train", {})),
        evaluation=EvaluationConfig(**payload.get("evaluation", {})),
    )
    config.validate()
    return config


def load_experiment_config(path: str | Path) -> tuple[ExperimentConfig, str, dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment configuration must be a JSON object.")
    return _construct(payload), canonical_fingerprint(payload), payload


def protocol_fingerprint(config: ExperimentConfig) -> str:
    """Fingerprint invariant only to the validation/official-test split switch."""

    payload = asdict(config)
    payload["data"]["test_policy"] = "FROZEN_SPLIT_SWITCH"
    return canonical_fingerprint(payload)
