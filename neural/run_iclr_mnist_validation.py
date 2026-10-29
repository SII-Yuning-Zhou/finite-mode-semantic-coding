"""Train, evaluate, or freeze one ICLR-aligned MNIST operating point."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from validation_iclr_mnist.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--action",
        choices=("train", "evaluate", "train-evaluate", "freeze"),
        default="train-evaluate",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint")
    arguments = parser.parse_args()
    config, fingerprint, _ = load_experiment_config(arguments.config)
    checkpoint = Path(arguments.checkpoint or Path(config.output_dir) / "final.pt")

    # Heavy imports remain below config validation so S0 never requires torch.
    from validation_iclr_mnist.evaluation import (
        evaluate_experiment,
        freeze_validation_protocol,
    )
    from validation_iclr_mnist.runtime import resolve_device
    from validation_iclr_mnist.training import train_experiment

    device = resolve_device(arguments.device)
    outputs: dict[str, object] = {}
    try:
        if arguments.action in {"train", "train-evaluate"}:
            outputs["training"] = train_experiment(
                config, config_fingerprint=fingerprint, device=device
            )
        if arguments.action in {"evaluate", "train-evaluate"}:
            outputs["evaluation"] = evaluate_experiment(
                config,
                config_fingerprint=fingerprint,
                checkpoint_path=checkpoint,
                device=device,
            )
        if arguments.action == "freeze":
            outputs["freeze"] = freeze_validation_protocol(
                config,
                config_fingerprint=fingerprint,
                checkpoint_path=checkpoint,
            )
    except Exception as exc:
        from validation_iclr_mnist.runtime import atomic_json

        atomic_json(
            Path(config.output_dir) / "status.json",
            {
                "state": "FAILED",
                "stage": arguments.action,
                "run_name": config.run_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(
        json.dumps(
            {
                "run_name": config.run_name,
                "action": arguments.action,
                "device": str(device),
                "outputs": outputs,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
