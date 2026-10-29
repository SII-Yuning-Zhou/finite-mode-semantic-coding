"""Deterministic multi-GPU-friendly grid launcher (one process per shard)."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path

from validation_iclr_mnist.protocol import COMPRESSAI_ENTROPY_PROTOCOL


def expand_grid(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = payload["base_config"]
    entries = []
    schedules = payload.get("schedules", [{"name": "continuous", "kind": "continuous", "counts": [1]}])
    for seed, rate_weight, schedule in itertools.product(
        payload["seeds"], payload["rate_weights"], schedules
    ):
        config = copy.deepcopy(base)
        schedule_name = schedule["name"]
        run_name = f"{payload['grid_name']}_s{seed}_lr{rate_weight}_{schedule_name}"
        config["run_name"] = run_name
        config["seed"] = seed
        config["train"]["rate_weight"] = rate_weight
        config["dither"] = {
            "kind": schedule["kind"],
            "counts": schedule["counts"],
            "codebook_seed": int(schedule.get("codebook_seed", 1729)),
            "schedule_seed": int(schedule.get("schedule_seed", 2718)),
        }
        config["output_dir"] = str(Path(payload["output_root"]) / run_name)
        entries.append(config)
    return entries


def evaluation_uses_current_bitstream(config: dict[str, object]) -> bool:
    report_path = Path(str(config["output_dir"])) / "evaluation_validation.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    rate_ledger = report.get("rate_ledger")
    try:
        from validation_iclr_mnist.config import canonical_fingerprint

        expected_fingerprint = canonical_fingerprint(config)
    except (ImportError, TypeError, ValueError):
        return False
    return (
        report.get("status") == "PASS"
        and report.get("config_fingerprint") == expected_fingerprint
        and isinstance(rate_ledger, dict)
        and rate_ledger.get("bitstream_protocol") == COMPRESSAI_ENTROPY_PROTOCOL
        and str(report.get("bitstream_roundtrip", "")).startswith("PASS")
    )


def plan_runs(
    entries: list[dict[str, object]], *, requested_action: str, resume: bool
) -> tuple[list[tuple[int, dict[str, object], str]], int]:
    tasks = []
    skipped = 0
    for index, config in enumerate(entries):
        if resume and evaluation_uses_current_bitstream(config):
            skipped += 1
            continue
        checkpoint = Path(str(config["output_dir"])) / "final.pt"
        action = "evaluate" if resume and checkpoint.is_file() else requested_action
        tasks.append((index, config, action))
    if resume:
        tasks.sort(key=lambda item: (item[2] == "evaluate", item[0]))
    return tasks, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--action", default="train-evaluate", choices=("train", "evaluate", "train-evaluate"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip current evaluations, evaluate existing final.pt files, and train only missing points.",
    )
    arguments = parser.parse_args()
    if arguments.shard_count < 1 or not 0 <= arguments.shard_index < arguments.shard_count:
        raise ValueError("Invalid shard index/count.")
    grid_path = Path(arguments.grid)
    entries = expand_grid(grid_path)
    tasks, skipped = plan_runs(
        entries, requested_action=arguments.action, resume=arguments.resume
    )
    selected = [
        task
        for task_index, task in enumerate(tasks)
        if task_index % arguments.shard_count == arguments.shard_index
    ]
    generated_root = Path("results/iclr_mnist/generated_configs") / grid_path.stem
    generated_root.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "grid": str(grid_path),
                "total": len(entries),
                "shard_index": arguments.shard_index,
                "shard_count": arguments.shard_count,
                "resume": arguments.resume,
                "pending": len(tasks),
                "skipped": skipped,
                "selected": len(selected),
            }
        ),
        flush=True,
    )
    for ordinal, (index, config, action) in enumerate(selected, start=1):
        config_path = generated_root / f"{index:04d}_{config['run_name']}.json"
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"[grid] shard={arguments.shard_index} run={ordinal}/{len(selected)} "
            f"global_index={index} action={action} name={config['run_name']}",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("run_iclr_mnist_validation.py")),
                "--config",
                str(config_path),
                "--action",
                action,
                "--device",
                arguments.device,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
