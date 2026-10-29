"""Model-M / Model-O receiver task heads on the frozen offset-state codec.

This module implements the receiver-side-information comparison requested for
the TCOM revision: on an unchanged encoder, entropy coder, and reconstruction
path, both task heads observe the 256-dimensional globally pooled features of
the frozen grayscale ResNet applied to the clipped decoded image. The model-M
head additionally receives a mean offset-index embedding; model O receives a
zero vector in that input slot. Neither head directly observes the decoded
latent or the quantized integer symbols.

Design constraints inherited from the v8.1 paired-codestream protocol:

- identical symbols, decoded latents, reconstructions, and labels for both
  heads within one (checkpoint, levels, schedule-repetition) cell;
- equal parameter counts: both heads own the same embedding table and linear
  classifier, and O simply zeroes the embedded state;
- shared initialization: O starts from M's parameters so any difference is
  learned usage of U, not an initialization artifact;
- validation cross-entropy estimates the population log loss of these
  restricted heads. Population log loss upper-bounds the Bayes conditional
  entropy for the observed features and optional pooled state. A finite
  validation estimate is not itself a guaranteed entropy bound; accuracy is
  a secondary statistic only.

This module deliberately avoids importing ``compressai`` so that its unit
tests run in any PyTorch environment; the worker script wires it to the
frozen codec.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class OffsetStateHead(nn.Module):
    """M/O heads with matched parameter shapes and a pooled offset-state input.

    The offset state U enters as per-coefficient level indexes.  An embedding
    table maps each level to a vector; the state summary is the mean over
    coefficient positions. This represents the state histogram and discards
    its spatial alignment. Independent sampling does not make this summary
    sufficient for the full state-conditioned task posterior. Model O receives
    a zero state summary alongside the same decoded-image features.
    """

    def __init__(
        self,
        feature_dimension: int,
        *,
        levels: int,
        embedding_dimension: int = 16,
        classes: int = 10,
        mode_aware: bool,
    ):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        self.mode_aware = mode_aware
        self.levels = levels
        self.embedding = nn.Embedding(levels, embedding_dimension)
        self.classifier = nn.Linear(feature_dimension + embedding_dimension, classes)

    def forward(
        self, features: torch.Tensor, level_indexes: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.embedding(level_indexes).mean(dim=1)
        if not self.mode_aware:
            embedded = torch.zeros_like(embedded)
        return self.classifier(torch.cat([features, embedded], dim=1))

    def state_table_bytes(self) -> int:
        """Deployment storage of the offset-state embedding table."""
        return self.embedding.weight.numel() * self.embedding.weight.element_size()


@dataclass(frozen=True)
class HeadMetrics:
    cross_entropy: float
    accuracy: float
    per_sample_nll_mean: float
    per_sample_nll_std: float


def make_head_pair(
    feature_dimension: int,
    *,
    levels: int,
    embedding_dimension: int = 16,
    classes: int = 10,
    seed: int,
    device: torch.device | str = "cpu",
) -> dict[str, OffsetStateHead]:
    """Build the M/O pair with shared initialization and equal capacity."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
    heads = {
        "model_m": OffsetStateHead(
            feature_dimension,
            levels=levels,
            embedding_dimension=embedding_dimension,
            classes=classes,
            mode_aware=True,
        ),
        "model_o": OffsetStateHead(
            feature_dimension,
            levels=levels,
            embedding_dimension=embedding_dimension,
            classes=classes,
            mode_aware=False,
        ),
    }
    for head in heads.values():
        for parameter in head.parameters():
            if parameter.dim() >= 2:
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5), generator=generator)
            # biases stay at their zero default for a deterministic shared init
    heads["model_o"].load_state_dict(heads["model_m"].state_dict())
    parameter_counts = {
        name: sum(p.numel() for p in head.parameters()) for name, head in heads.items()
    }
    if parameter_counts["model_m"] != parameter_counts["model_o"]:
        raise RuntimeError("M/O heads must have equal parameter counts.")
    return {name: head.to(device) for name, head in heads.items()}


def train_head(
    head: OffsetStateHead,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    device: torch.device | str,
) -> None:
    """Train one head with plain cross-entropy on frozen features."""

    features, level_indexes, labels = train_data
    dataset = TensorDataset(features, level_indexes, labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF),
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    head.train()
    for _ in range(epochs):
        for batch_features, batch_levels, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_levels = batch_levels.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                head(batch_features, batch_levels), batch_labels
            )
            loss.backward()
            optimizer.step()
    head.eval()


@torch.no_grad()
def evaluate_head(
    head: OffsetStateHead,
    validation_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    device: torch.device | str,
) -> HeadMetrics:
    """Validation cross-entropy in nats and classification accuracy."""

    features, level_indexes, labels = validation_data
    logits = head(features.to(device), level_indexes.to(device))
    per_sample_nll = F.cross_entropy(logits, labels.to(device), reduction="none")
    prediction = logits.argmax(dim=1).cpu()
    return HeadMetrics(
        cross_entropy=float(per_sample_nll.mean()),
        accuracy=float((prediction == labels).double().mean()),
        per_sample_nll_mean=float(per_sample_nll.mean()),
        per_sample_nll_std=float(per_sample_nll.std(unbiased=True)),
    )


def run_paired_mo(
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    validation_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    levels: int,
    head_seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device | str,
    embedding_dimension: int = 16,
) -> dict[str, object]:
    """Train and evaluate the equal-capacity M/O pair on identical data."""

    feature_dimension = train_data[0].shape[1]
    heads = make_head_pair(
        feature_dimension,
        levels=levels,
        embedding_dimension=embedding_dimension,
        seed=head_seed,
        device=device,
    )
    report: dict[str, object] = {}
    for name, head in heads.items():
        train_head(
            head,
            train_data,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=head_seed,
            device=device,
        )
        metrics = evaluate_head(head, validation_data, device=device)
        report[name] = {
            "parameters": sum(p.numel() for p in head.parameters()),
            "state_table_bytes": head.state_table_bytes(),
            "cross_entropy": metrics.cross_entropy,
            "accuracy": metrics.accuracy,
            "per_sample_nll_std": metrics.per_sample_nll_std,
        }
    m = report["model_m"]
    o = report["model_o"]
    assert isinstance(m, dict) and isinstance(o, dict)
    report["paired_difference_m_minus_o"] = {
        "cross_entropy": m["cross_entropy"] - o["cross_entropy"],
        "accuracy": m["accuracy"] - o["accuracy"],
    }
    return report
