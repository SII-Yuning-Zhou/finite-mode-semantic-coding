"""Run the lightweight, package-relative reproduction path without neural training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def run_step(name, argv, output):
    start = time.perf_counter()
    process = subprocess.run([sys.executable, *argv], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=240)
    elapsed = time.perf_counter() - start
    stdout, stderr = process.stdout, process.stderr
    record = {"name": name, "argv": ["python", *argv],
              "seconds": round(elapsed, 4), "returncode": process.returncode,
              "status": "PASS" if process.returncode == 0 else "FAIL"}
    test_count = re.search(r"Ran (\d+) tests?", stderr)
    skips = re.search(r"OK \(skipped=(\d+)\)", stderr)
    if test_count:
        record["tests_run"] = int(test_count.group(1))
    if skips:
        record["skipped"] = int(skips.group(1))
        record["status"] = "SKIP" if record.get("tests_run") == record["skipped"] else "PASS_WITH_SKIPS"
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        record["result"] = parsed
    # Derived command outputs are retained separately from the archived evidence.
    (output / f"{name}.stdout.txt").write_text(stdout, encoding="utf-8")
    (output / f"{name}.stderr.txt").write_text(stderr, encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("name", "status", "seconds")}), flush=True)
    if process.returncode:
        print(stderr[-4000:], file=sys.stderr)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--tests", action="store_true", help="also run fast regression tests; optional ANS tests may skip")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if output == ROOT or any((ROOT / part) == output or (ROOT / part) in output.parents
                             for part in ("data", "results", "theory", "neural", "figures", "configs", "docs", "reproduce", "tests")):
        raise ValueError("Choose a derived-output directory, not a source/evidence directory")
    output.mkdir(parents=True, exist_ok=True)
    steps = [
        ("frozen_artifacts", ["neural/verify_frozen_artifacts.py", "--output", str(output / "frozen_artifacts_validation.json")]),
        ("theory", ["-m", "theory.verify", "--output", str(output / "theory_verification.json")]),
        ("neural_records", ["neural/quick_verify.py"]),
        ("figure_sources", ["-m", "reproduce.check_data", "--output", str(output / "figure_source_checks.json")]),
        ("tables", ["-m", "reproduce.tables", "--output", str(output / "tables")]),
    ]
    if not args.skip_plots:
        steps.append(("figures", ["-m", "reproduce.figures", "--output", str(output / "figures")]))
    if args.tests:
        steps.extend([
            ("reproduction_tests", ["-m", "unittest", "discover", "-s", "tests/reproduce", "-v"]),
            ("theory_tests", ["-m", "unittest", "discover", "-s", "tests/theory", "-v"]),
            ("neural_config_tests", ["-m", "unittest", "discover", "-s", "neural", "-p", "test_release_config.py", "-v"]),
            ("classifier_gate_tests", ["-m", "unittest", "discover", "-s", "neural", "-p", "test_iclr_mnist_classifier_gate.py", "-v"]),
            ("ans_tests", ["-m", "unittest", "discover", "-s", "neural", "-p", "test_iclr_mnist_conditional_rate_v8.py", "-v"]),
        ])
    results = []
    for name, argv in steps:
        record = run_step(name, argv, output)
        results.append(record)
        if record["returncode"]:
            break
    failed = any(x["returncode"] for x in results)
    report = {
        "status": "FAIL" if failed else ("PASS_WITH_SKIPS" if any(x["status"] in ("SKIP", "PASS_WITH_SKIPS") for x in results) else "PASS"),
        "scope": "Frozen-artifact integrity, theory/archive arithmetic, original-profile reaggregation and portable plotting; not neural training or latency measurement replay",
        "python": sys.version.split()[0], "steps": results,
        "additional_artifacts": "Original frozen neural weights and endpoint profiles are included; the standard-library path does not execute the checkpoints",
    }
    (output / "reproduction_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output / "reproduction_report.json")}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
