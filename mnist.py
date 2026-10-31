"""Public command-line interface for the MNIST codec experiments.

The launcher uses only the Python standard library. Neural dependencies are
needed only for commands that execute a model. Frozen configurations and
checkpoint locations remain compatible with the archived research artifacts.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
RATES = (0, 200, 5000, 50000)
LEVELS = (1, 3, 7, 15, 31)
GRID = Path("configs/iclr_mnist/grid_s1_rate_calibration_v7.json")
CLASSIFIER_CONFIG = Path("configs/iclr_mnist/classifier_reference_v1.json")
MODEL_PATHS = {
    rate: Path(f"results/iclr_mnist/s1_rate_calibration_v7/s1_rate_calibration_v7_s20260811_lr{rate}_continuous/final.pt")
    for rate in RATES
}
CLASSIFIER_PATH = Path("results/iclr_mnist/classifiers/iclr_gray_resnet18_small_v2.pt")
ENGINES = {
    "verify": "verify_frozen_artifacts.py",
    "checkpoints": "check_frozen_checkpoints.py",
    "evaluate": "audit_iclr_mnist_conditional_rate_v8.py",
    "profile": "profile_iclr_mnist_edge_deployment.py",
    "train": "run_iclr_mnist_validation.py",
    "prepare-classifier": "prepare_iclr_mnist_classifier.py",
}


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="mnist.py", description=__doc__.split("\n\n", 1)[0])
    commands = result.add_subparsers(dest="command", required=True, metavar="COMMAND")
    models = commands.add_parser("list-models", help="show the four bundled codec operating points")
    models.add_argument("--json", action="store_true", help="write a machine-readable model inventory")
    verify = commands.add_parser("verify", help="check bundled weights, coding records and original endpoint profiles")
    verify.add_argument("--output", type=Path, default=Path("outputs/mnist/frozen_artifacts_validation.json"), metavar="FILE")
    checkpoints = commands.add_parser("checkpoints", help="safely load the frozen models and optionally run CPU smoke checks")
    checkpoints.add_argument("--run-cpu-smoke", action="store_true", help="also test tiny synthetic forward, coding and frame round trips")
    checkpoints.add_argument("--output", type=Path, default=Path("outputs/checkpoint_validation.json"), metavar="FILE")
    evaluate = commands.add_parser("evaluate", help="evaluate marginal and conditional coding with a frozen codec")
    evaluate.add_argument("--rate-weight", type=int, choices=RATES, required=True, metavar="RATE")
    evaluate.add_argument("--grid", type=Path, metavar="FILE", help="custom operating-point grid; uses the bundled grid by default")
    evaluate.add_argument("--device", default="cpu", help="execution device, e.g. cpu or cuda:0")
    evaluate.add_argument("--levels", nargs="+", type=positive_int, default=list(LEVELS), metavar="L")
    evaluate.add_argument("--schedule-repetition", type=nonnegative_int, default=0, metavar="N")
    evaluate.add_argument("--maximum-samples", type=positive_int, default=5000, metavar="N")
    evaluate.add_argument("--batch-size", type=positive_int, default=512, metavar="N")
    evaluate.add_argument("--output", type=Path, metavar="FILE", help="report file; defaults to outputs/mnist/rate_RATE_rep_N.json")
    evaluate.epilog = "Uses the published 16-bit CDF, [-128,128] symbol support and escape-tail protocol. Include L=1 as the identity control."
    profile = commands.add_parser("profile", help="measure batch-one endpoint storage and encode/decode latency")
    profile.add_argument("--rate-weight", type=int, choices=RATES, required=True, metavar="RATE")
    profile.add_argument("--grid", type=Path, metavar="FILE", help="custom operating-point grid; uses the bundled grid by default")
    profile.add_argument("--checkpoint", type=Path, metavar="FILE", help="optional explicit frozen checkpoint")
    profile.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    profile.add_argument("--threads", type=positive_int, default=1, metavar="N")
    profile.add_argument("--levels", nargs="+", type=positive_int, default=list(LEVELS), metavar="L")
    profile.add_argument("--samples", type=positive_int, default=128, metavar="N")
    profile.add_argument("--warmup", type=nonnegative_int, default=20, metavar="N")
    profile.add_argument("--repetitions", type=positive_int, default=100, metavar="N")
    profile.add_argument("--output-dir", type=Path, default=Path("outputs/edge_profile"), metavar="DIR")
    train = commands.add_parser("train", help="start new training runs in a separate working directory")
    train.add_argument("--grid", type=Path, metavar="FILE", help="custom training grid; uses the bundled settings by default")
    train.add_argument("--rate-weight", type=int, choices=RATES, metavar="RATE", help="train one operating point instead of all four")
    train.add_argument("--device", default="auto", help="execution device, e.g. auto or cuda:0")
    train.add_argument("--shard-index", type=nonnegative_int, default=0, metavar="N")
    train.add_argument("--shard-count", type=positive_int, default=1, metavar="N")
    train.add_argument("--resume", action="store_true", help="evaluate existing new-run checkpoints instead of retraining them")
    train.add_argument("--output-dir", type=Path, default=Path("runs/mnist_training"), metavar="DIR")
    prepare = commands.add_parser("prepare-classifier", help="train a new frozen evaluator without replacing the bundled classifier")
    prepare.add_argument("--config", type=Path, metavar="FILE", help="custom classifier configuration; uses the bundled reference by default")
    prepare.add_argument("--device", default="auto", help="execution device, e.g. auto or cuda:0")
    prepare.add_argument("--output", type=Path, default=Path("runs/classifier/mnist_classifier.pt"), metavar="FILE")
    return result


def resolve_input(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root/path).resolve()


def resolve_output(path: Path, root: Path) -> Path:
    target = resolve_input(path, root)
    root = root.resolve()
    if target == root or (target.is_relative_to(root) and target.relative_to(root).parts[0] not in ("outputs", "runs")):
        raise ValueError("Choose an outputs/ or runs/ location, not a source, model or archived-evidence directory.")
    return target


def engine_command(name: str, root: Path) -> list[str]:
    return [sys.executable, str(root/"neural"/ENGINES[name])]


def require_neural_dependencies() -> None:
    missing = [name for name in ("torch", "torchvision", "compressai") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Optional neural dependencies are missing: " + ", ".join(missing)
                           + ". Install a compatible PyTorch/torchvision build and neural/requirements.txt first.")


def command_for(args: argparse.Namespace, root: Path = ROOT) -> list[str]:
    """Build an engine invocation without importing it or creating files."""
    command = engine_command(args.command, root)
    if args.command in ("verify", "checkpoints"):
        command += ["--output", str(resolve_output(args.output, root))]
        if args.command == "checkpoints":
            command += ["--checkpoint-root", str(root)]
            if args.run_cpu_smoke:
                command.append("--run-cpu-smoke")
    elif args.command == "evaluate":
        if 1 not in args.levels or len(set(args.levels)) != len(args.levels):
            raise ValueError("Choose unique offset levels including L=1.")
        output = args.output or Path(f"outputs/mnist/rate_{args.rate_weight}_rep_{args.schedule_repetition}.json")
        command += ["--grid", str(resolve_input(args.grid or GRID, root)),
                    "--rate-weight", str(args.rate_weight), "--levels", *map(str, args.levels),
                    "--maximum-samples", str(args.maximum_samples), "--batch-size", str(args.batch_size),
                    "--symbol-minimum", "-128", "--symbol-maximum", "128", "--precision", "16", "--escape-tail",
                    "--schedule-repetition", str(args.schedule_repetition), "--device", args.device,
                    "--output", str(resolve_output(output, root))]
    elif args.command == "profile":
        if len(set(args.levels)) != len(args.levels):
            raise ValueError("Choose unique offset levels.")
        command += ["--grid", str(resolve_input(args.grid or GRID, root)),
                    "--rate-weight", str(args.rate_weight), "--device", args.device,
                    "--threads", str(args.threads), "--levels", *map(str, args.levels),
                    "--samples", str(args.samples), "--warmup", str(args.warmup),
                    "--repetitions", str(args.repetitions), "--output-dir", str(resolve_output(args.output_dir, root))]
        if args.checkpoint:
            command += ["--checkpoint", str(resolve_input(args.checkpoint, root))]
    elif args.command == "prepare-classifier":
        command += ["--config", str(resolve_input(args.config or CLASSIFIER_CONFIG, root)),
                    "--output", str(resolve_output(args.output, root)), "--device", args.device]
    else:
        raise ValueError("This command does not have a single-engine invocation.")
    return command


def model_inventory(root: Path = ROOT) -> dict:
    def record(model_id, relative, rate=None):
        path = root/relative
        value = {"model":model_id, "available":path.is_file()}
        if rate is not None:
            value["rate_weight"] = rate
        if path.is_file():
            value["checkpoint_bytes"] = path.stat().st_size
        return value
    return {"codecs":[record(f"mnist-rate-{rate}", path, rate) for rate,path in MODEL_PATHS.items()],
            "classifier":record("mnist-classifier-v2", CLASSIFIER_PATH)}


def training_plan(args: argparse.Namespace, root: Path = ROOT) -> list[tuple[Path, dict, str]]:
    """Derive new-run configs while leaving the bundled grid and weights intact."""
    if args.shard_index >= args.shard_count:
        raise ValueError("shard-index must be smaller than shard-count.")
    output = resolve_output(args.output_dir, root)
    grid = json.loads(resolve_input(args.grid or GRID, root).read_text(encoding="utf-8"))
    rates = grid["rate_weights"] if args.rate_weight is None else [args.rate_weight]
    if any(rate not in grid["rate_weights"] for rate in rates):
        raise ValueError("The selected operating point is absent from the training grid.")
    schedules = grid.get("schedules", [{"name":"continuous", "kind":"continuous", "counts":[1]}])
    tasks = []
    for index, (seed, rate, schedule) in enumerate(itertools.product(grid["seeds"], rates, schedules)):
        if index % args.shard_count != args.shard_index:
            continue
        config = copy.deepcopy(grid["base_config"])
        run_name = f"mnist_s{seed}_rate{rate}_{schedule['name']}"
        config.update(run_name=run_name, seed=seed, output_dir=str(output/run_name))
        config["train"]["rate_weight"] = rate
        config["dither"] = {"kind":schedule["kind"], "counts":schedule["counts"],
                            "codebook_seed":int(schedule.get("codebook_seed",1729)),
                            "schedule_seed":int(schedule.get("schedule_seed",2718))}
        checkpoint = Path(config["output_dir"])/"final.pt"
        if checkpoint.exists() and not args.resume:
            raise FileExistsError("A new-run checkpoint already exists. Use --resume or choose a new output directory.")
        tasks.append((output/"configs"/(run_name+".json"), config,
                      "evaluate" if checkpoint.is_file() and args.resume else "train-evaluate"))
    return tasks


def run_training(args: argparse.Namespace, root: Path = ROOT) -> int:
    tasks = training_plan(args, root)
    for ordinal, (config_path, config, action) in enumerate(tasks, start=1):
        payload = json.dumps(config,indent=2)+"\n"
        config_path.parent.mkdir(parents=True,exist_ok=True)
        if config_path.exists():
            if config_path.read_text(encoding="utf-8") != payload:
                raise FileExistsError("An existing new-run configuration differs; choose a new output directory.")
        else:
            with config_path.open("x",encoding="utf-8") as stream:
                stream.write(payload)
        print(f"[train] run {ordinal}/{len(tasks)}: {config['run_name']} ({action})",flush=True)
        result = subprocess.run(engine_command("train",root)+["--config",str(config_path),
                                "--action",action,"--device",args.device],cwd=root)
        if result.returncode:
            return result.returncode
    return 0


def main(argv=None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        if args.command == "list-models":
            inventory = model_inventory()
            if args.json:
                print(json.dumps(inventory,indent=2))
            else:
                print("Model                      Rate weight   Available")
                for value in inventory["codecs"]:
                    print(f"{value['model']:<27}{value['rate_weight']:<14}{'yes' if value['available'] else 'no'}")
                print(f"{inventory['classifier']['model']:<27}{'-':<14}{'yes' if inventory['classifier']['available'] else 'no'}")
            return 0
        if args.command != "verify":
            require_neural_dependencies()
        if args.command == "train":
            return run_training(args)
        return subprocess.run(command_for(args),cwd=ROOT).returncode
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError) as exc:
        print("Error: "+str(exc),file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
