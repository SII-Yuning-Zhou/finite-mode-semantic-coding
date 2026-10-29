"""Pretrain once and freeze the clean-MNIST evaluator used by every grid point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from validation_iclr_mnist.classifier_gate import (
    classifier_gate_rule,
    classifier_health_failures,
)
from validation_iclr_mnist.config import load_experiment_config
from validation_iclr_mnist.runtime import atomic_json, atomic_torch_save, resolve_device, seed_everything
from validation_iclr_mnist.training import build_loaders, build_models, pretrain_classifier


@torch.no_grad()
def classifier_diagnostics(classifier, loader, device: torch.device) -> dict[str, float | int]:
    classifier.eval()
    total = 0
    clean_correct = 0
    degraded_correct = 0
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        clean = batch["hr"].to(device, non_blocking=True)
        degraded = batch["x"].to(device, non_blocking=True)
        clean_correct += int((classifier(clean).argmax(1) == labels).sum())
        degraded_correct += int((classifier(degraded).argmax(1) == labels).sum())
        total += len(labels)
    if total == 0:
        raise RuntimeError("Classifier diagnostic loader is empty.")
    return {
        "validation_samples": total,
        "clean_accuracy": clean_correct / total,
        "degraded_7x7_accuracy": degraded_correct / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    config, fingerprint, _ = load_experiment_config(arguments.config)
    if config.train.classifier_checkpoint is not None:
        raise ValueError("Preparation config must not point to an existing classifier.")
    device = resolve_device(arguments.device)
    seed_everything(config.seed)
    _, source_loader, _, validation_loader = build_loaders(config)
    _, classifier, _ = build_models(config, device)
    history = pretrain_classifier(classifier, source_loader, config, device)
    diagnostics = classifier_diagnostics(classifier, validation_loader, device)
    health_failures = classifier_health_failures(diagnostics)
    if health_failures:
        raise RuntimeError(
            "Frozen classifier failed its target-domain health Gate: "
            + "; ".join(health_failures)
        )
    gate_rule = classifier_gate_rule()
    output = Path(arguments.output)
    atomic_torch_save(
        output,
        {
            "schema_version": 1,
            "seed": config.seed,
            "config_fingerprint": fingerprint,
            "classifier": classifier.state_dict(),
            "history": history,
            "diagnostics": diagnostics,
            "classifier_gate": gate_rule,
        },
    )
    report = {
        "status": "PASS",
        "output": str(output.resolve()),
        "seed": config.seed,
        "history": history,
        "diagnostics": diagnostics,
        "classifier_gate": gate_rule,
        "parameters": sum(parameter.numel() for parameter in classifier.parameters()),
        "torch": torch.__version__,
    }
    atomic_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
