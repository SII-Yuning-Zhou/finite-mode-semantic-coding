"""Aggregate the escape-aware, repeated-schedule conditional-rate audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from validation_iclr_mnist.runtime import atomic_json


T95_DF4 = 2.7764451051977987


def mean_ci95(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = (
        T95_DF4 * standard_deviation / math.sqrt(len(values))
        if len(values) == 5
        else float("nan")
    )
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
    }


def requested_levels(report: dict[str, object]) -> set[int]:
    return {int(level) for level in report.get("levels", [])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-repetitions", type=int, default=5)
    parser.add_argument("--minimum-actual-gain-bpp", type=float, default=0.001)
    arguments = parser.parse_args()

    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in arguments.inputs]
    failures: list[str] = []
    expected_reports = 4 * arguments.expected_repetitions
    if len(reports) != expected_reports:
        failures.append(f"Expected {expected_reports} reports, found {len(reports)}.")
    if any(report.get("status") != "PASS" for report in reports):
        failures.append("At least one checkpoint/schedule report failed its technical Gate.")
    if any(report.get("audit_protocol") != "UQ-CONDITIONAL-RANS-ESCAPE-V2" for report in reports):
        failures.append("At least one input did not use the escape-aware v8.1 protocol.")
    expected_levels = {1, 3, 7, 15, 31}
    if any(requested_levels(report) != expected_levels for report in reports):
        failures.append("At least one report has an incomplete offset-level set.")

    keys = [(float(r["rate_weight"]), int(r["schedule_repetition"])) for r in reports]
    if len(set(keys)) != len(keys):
        failures.append("Checkpoint/schedule input keys are not unique.")
    expected_rates = {0.0, 200.0, 5000.0, 50000.0}
    if {rate for rate, _ in keys} != expected_rates:
        failures.append("The four frozen checkpoint weights are incomplete.")
    for rate in expected_rates:
        repetitions = {rep for current, rep in keys if current == rate}
        if repetitions != set(range(arguments.expected_repetitions)):
            failures.append(f"rate_weight={rate:g} has an incomplete repetition set.")

    grouped: dict[tuple[float, int], list[dict[str, object]]] = defaultdict(list)
    for report in reports:
        rate = float(report["rate_weight"])
        for point in report["points"]:
            grouped[(rate, int(point["levels"]))].append(point)
    expected_group_keys = {
        (rate, level) for rate in expected_rates for level in expected_levels
    }
    if set(grouped) != expected_group_keys:
        failures.append("The checkpoint/offset-level Cartesian product is incomplete.")

    summaries: list[dict[str, object]] = []
    mean_actual_checks = 0
    mean_quantized_checks = 0
    total_checks = 0
    endpoint_means: list[float] = []
    maximum_levels = max(expected_levels)
    for (rate, levels), points in sorted(grouped.items()):
        if len(points) != arguments.expected_repetitions:
            failures.append(
                f"rate_weight={rate:g}, L={levels} has {len(points)} repetitions."
            )
        actual = [float(p["actual_conditioning_gain_bpp"]) for p in points]
        quantized = [float(p["quantized_cdf_conditioning_gain_bpp"]) for p in points]
        actual_summary = mean_ci95(actual)
        quantized_summary = mean_ci95(quantized)
        if levels == 1:
            if any(abs(value) > 1e-12 for value in actual):
                failures.append(f"L=1 identity failed at rate_weight={rate:g}.")
        else:
            total_checks += 1
            if actual_summary["mean"] >= -1e-6:
                mean_actual_checks += 1
            if quantized_summary["mean"] >= -1e-8:
                mean_quantized_checks += 1
        if levels == maximum_levels:
            endpoint_means.append(actual_summary["mean"])
        summaries.append({
            "rate_weight": rate,
            "levels": levels,
            "repetitions": len(points),
            "actual_gain_bpp": actual_summary,
            "quantized_cdf_gain_bpp": quantized_summary,
            "all_same_symbols_and_reconstruction": all(
                bool(p["same_symbols_and_reconstruction"]) for p in points
            ),
            "maximum_escape_tail_mass": max(float(p["max_escape_tail_mass"]) for p in points),
            "maximum_normalization_residual": max(
                float(p["max_support_normalization_residual"]) for p in points
            ),
        })

    median_endpoint = statistics.median(endpoint_means) if endpoint_means else 0.0
    scientific_failures = [
        f"Technical prerequisite failed: {failure}" for failure in failures
    ]
    if mean_actual_checks != total_checks:
        scientific_failures.append(
            f"Only {mean_actual_checks}/{total_checks} L>1 checkpoint-level means "
            "had nonnegative real-payload gain."
        )
    if mean_quantized_checks != total_checks:
        scientific_failures.append(
            f"Only {mean_quantized_checks}/{total_checks} L>1 checkpoint-level means "
            "had nonnegative quantized-CDF gain."
        )
    if median_endpoint < arguments.minimum_actual_gain_bpp:
        scientific_failures.append(
            f"Median L={maximum_levels} checkpoint-mean gain {median_endpoint:.6g} bpp "
            f"is below {arguments.minimum_actual_gain_bpp:.6g} bpp."
        )
    scientific_pass = not scientific_failures
    aggregate = {
        "schema_version": 2,
        "status": "PASS" if scientific_pass else "FAIL",
        "audit_protocol": "UQ-CONDITIONAL-RANS-ESCAPE-V2",
        "technical_gate": {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
        },
        "scientific_gate": {
            "status": "PASS" if scientific_pass else "FAIL",
            "failures": scientific_failures,
            "claim_scope": "operational conditional-entropy bridge on frozen MNIST codecs",
            "scope_exclusions": [
                "not a proof of the information-theoretic theorem",
                "does not manipulate the finite whole-codec mode budget K",
                "does not identify a general K-Rc boundary",
                "does not equate a stored PRNG seed with iid schedule entropy H(U)",
            ],
            "maximum_levels": maximum_levels,
            "checkpoint_mean_endpoint_gains_bpp": endpoint_means,
            "median_checkpoint_mean_endpoint_gain_bpp": median_endpoint,
            "minimum_actual_gain_bpp": arguments.minimum_actual_gain_bpp,
            "nonnegative_actual_mean_checks": mean_actual_checks,
            "nonnegative_quantized_mean_checks": mean_quantized_checks,
            "total_mean_checks": total_checks,
        },
        "summaries": summaries,
        "inputs": arguments.inputs,
    }
    atomic_json(arguments.output, aggregate)
    print(json.dumps({
        "status": aggregate["status"],
        "technical_gate": aggregate["technical_gate"],
        "scientific_gate": aggregate["scientific_gate"],
    }, indent=2))
    # A scientifically negative result is still a valid completed experiment
    # and must be packaged for audit.  Input/JSON/runtime errors already raise;
    # Gate FAIL is represented in the output rather than by discarding it.


if __name__ == "__main__":
    main()
