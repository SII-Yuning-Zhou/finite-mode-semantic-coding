"""Audit and visualize the completed conditional-rate v8.1 evidence package."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPORT_PATTERN = re.compile(r"rate_(0|200|5000|50000)_rep_([0-4])\.json")
RATES = (0.0, 200.0, 5000.0, 50000.0)
LEVELS = (1, 3, 7, 15, 31)
T95_DF4 = 2.7764451051977987


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_ci95(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half_width = T95_DF4 * standard_deviation / math.sqrt(len(values))
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "ci95_half_width": half_width,
    }


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=tolerance)


def audit_and_collect(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    reports: list[dict[str, object]] = []
    report_paths = sorted(
        path for path in root.iterdir() if REPORT_PATTERN.fullmatch(path.name)
    )
    if len(report_paths) != 20:
        raise RuntimeError(f"Expected 20 reports, found {len(report_paths)}.")

    aggregate = read_json(root / "aggregate.json")
    completion = read_json(root / "completion.json")
    provenance = read_json(root / "provenance.json")
    if aggregate["status"] != "PASS":
        raise RuntimeError("Aggregate status is not PASS.")
    if aggregate["technical_gate"]["status"] != "PASS":
        raise RuntimeError("Technical Gate is not PASS.")
    if aggregate["scientific_gate"]["status"] != "PASS":
        raise RuntimeError("Scientific Gate is not PASS.")
    if completion["status"] != "COMPLETE":
        raise RuntimeError("Completion record is not COMPLETE.")
    if provenance["protocol"] != "UQ-CONDITIONAL-RANS-ESCAPE-V2":
        raise RuntimeError("Unexpected provenance protocol.")

    config_fingerprints: dict[float, set[str]] = {rate: set() for rate in RATES}
    checkpoint_fingerprints: dict[float, set[str]] = {rate: set() for rate in RATES}
    raw_rows: list[dict[str, object]] = []
    report_keys: set[tuple[float, int]] = set()
    for path in report_paths:
        report = read_json(path)
        reports.append(report)
        rate = float(report["rate_weight"])
        repetition = int(report["schedule_repetition"])
        key = (rate, repetition)
        if key in report_keys:
            raise RuntimeError(f"Duplicate report key {key}.")
        report_keys.add(key)
        if report["status"] != "PASS":
            raise RuntimeError(f"Report did not pass: {path.name}")
        if report["audit_protocol"] != "UQ-CONDITIONAL-RANS-ESCAPE-V2":
            raise RuntimeError(f"Protocol mismatch: {path.name}")
        if report["split"] != "validation" or int(report["samples"]) != 5000:
            raise RuntimeError(f"Unexpected evaluation split/samples: {path.name}")
        if tuple(report["levels"]) != LEVELS or int(report["completed_levels"]) != 5:
            raise RuntimeError(f"Incomplete level set: {path.name}")
        if tuple(report["latent_shape"]) != (12, 4, 4):
            raise RuntimeError(f"Latent shape mismatch: {path.name}")
        if len(report["points"]) != 5:
            raise RuntimeError(f"Point count mismatch: {path.name}")
        config_fingerprints[rate].add(str(report["config_fingerprint"]))
        checkpoint_fingerprints[rate].add(
            str(report["checkpoint_protocol_fingerprint"])
        )
        for point in report["points"]:
            level = int(point["levels"])
            if point["status"] != "PASS":
                raise RuntimeError(f"Level did not pass: {path.name}, L={level}")
            if point["bitstream_roundtrip"] != "PASS_BIT_EXACT":
                raise RuntimeError(f"Round trip failed: {path.name}, L={level}")
            if not bool(point["same_symbols_and_reconstruction"]):
                raise RuntimeError(f"Symbols/reconstruction differ: {path.name}, L={level}")
            if not bool(point["escape_tail_cdf_and_bypass"]):
                raise RuntimeError(f"Escape coding disabled: {path.name}, L={level}")
            if float(point["max_support_normalization_residual"]) > 1e-6:
                raise RuntimeError(f"CDF residual failed: {path.name}, L={level}")
            if level == 1 and abs(float(point["actual_conditioning_gain_bpp"])) > 1e-12:
                raise RuntimeError(f"L=1 identity failed: {path.name}")
            raw_rows.append(
                {
                    "rate_weight": rate,
                    "repetition": repetition,
                    "levels": level,
                    "offset_entropy_bits_per_coefficient": float(
                        point["theoretical_iid_schedule_entropy_bits_per_coefficient"]
                    ),
                    "actual_gain_bpp": float(point["actual_conditioning_gain_bpp"]),
                    "quantized_gain_bpp": float(
                        point["quantized_cdf_conditioning_gain_bpp"]
                    ),
                    "floating_gain_bpp": float(
                        point["floating_model_conditioning_gain_bpp"]
                    ),
                    "conditional_actual_bpp": float(point["conditional_actual_bpp"]),
                    "marginal_actual_bpp": float(point["marginal_actual_bpp"]),
                    "conditional_table_bytes": int(point["conditional_table_bytes"]),
                    "marginal_table_bytes": int(point["marginal_table_bytes"]),
                    "table_byte_ratio": float(point["table_byte_ratio"]),
                    "mse_x": float(point["mse_x"]),
                    "mse_hr": float(point["mse_hr"]),
                    "accuracy": float(point["accuracy"]),
                    "max_tail_mass": float(point["max_escape_tail_mass"]),
                    "normalization_residual": float(
                        point["max_support_normalization_residual"]
                    ),
                    "conditional_codec_seconds": float(
                        point["conditional_codec_seconds"]
                    ),
                    "marginal_codec_seconds": float(point["marginal_codec_seconds"]),
                }
            )

    expected_keys = {(rate, repetition) for rate in RATES for repetition in range(5)}
    if report_keys != expected_keys:
        raise RuntimeError("Checkpoint/repetition Cartesian product is incomplete.")
    if any(len(values) != 1 for values in config_fingerprints.values()):
        raise RuntimeError("A checkpoint uses inconsistent config fingerprints.")
    if any(len(values) != 1 for values in checkpoint_fingerprints.values()):
        raise RuntimeError("A checkpoint uses inconsistent protocol fingerprints.")

    grouped: dict[tuple[float, int], list[dict[str, object]]] = {}
    for row in raw_rows:
        grouped.setdefault((float(row["rate_weight"]), int(row["levels"])), []).append(row)
    if len(grouped) != 20 or any(len(values) != 5 for values in grouped.values()):
        raise RuntimeError("Every checkpoint-level cell must contain five schedules.")

    summary_rows: list[dict[str, object]] = []
    for (rate, level), rows in sorted(grouped.items()):
        actual = mean_ci95([float(row["actual_gain_bpp"]) for row in rows])
        quantized = mean_ci95([float(row["quantized_gain_bpp"]) for row in rows])
        summary_rows.append(
            {
                "rate_weight": rate,
                "levels": level,
                "offset_entropy_bits_per_coefficient": math.log2(level),
                **{f"actual_gain_{key}": value for key, value in actual.items()},
                **{f"quantized_gain_{key}": value for key, value in quantized.items()},
                "conditional_actual_bpp_mean": statistics.fmean(
                    float(row["conditional_actual_bpp"]) for row in rows
                ),
                "marginal_actual_bpp_mean": statistics.fmean(
                    float(row["marginal_actual_bpp"]) for row in rows
                ),
                "conditional_table_bytes": int(rows[0]["conditional_table_bytes"]),
                "marginal_table_bytes": int(rows[0]["marginal_table_bytes"]),
                "table_byte_ratio": float(rows[0]["table_byte_ratio"]),
                "mse_x_mean": statistics.fmean(float(row["mse_x"]) for row in rows),
                "mse_hr_mean": statistics.fmean(float(row["mse_hr"]) for row in rows),
                "accuracy_mean": statistics.fmean(float(row["accuracy"]) for row in rows),
                "maximum_tail_mass": max(float(row["max_tail_mass"]) for row in rows),
            }
        )

    aggregate_lookup = {
        (float(item["rate_weight"]), int(item["levels"])): item
        for item in aggregate["summaries"]
    }
    for summary in summary_rows:
        key = (float(summary["rate_weight"]), int(summary["levels"]))
        stored = aggregate_lookup[key]
        for metric in ("mean", "standard_deviation", "ci95_lower", "ci95_upper"):
            if not close(
                float(summary[f"actual_gain_{metric}"]),
                float(stored["actual_gain_bpp"][metric]),
            ):
                raise RuntimeError(f"Aggregate actual statistic mismatch at {key}: {metric}")
            if not close(
                float(summary[f"quantized_gain_{metric}"]),
                float(stored["quantized_cdf_gain_bpp"][metric]),
            ):
                raise RuntimeError(f"Aggregate proxy statistic mismatch at {key}: {metric}")

    endpoint_means = [
        float(row["actual_gain_mean"])
        for row in summary_rows
        if int(row["levels"]) == 31
    ]
    endpoint_median = statistics.median(endpoint_means)
    stored_median = float(
        aggregate["scientific_gate"]["median_checkpoint_mean_endpoint_gain_bpp"]
    )
    if not close(endpoint_median, stored_median):
        raise RuntimeError("Endpoint median does not reproduce the aggregate.")

    lgt1 = [row for row in raw_rows if int(row["levels"]) > 1]
    proxy = np.array([float(row["quantized_gain_bpp"]) for row in lgt1])
    actual = np.array([float(row["actual_gain_bpp"]) for row in lgt1])
    correlation = float(np.corrcoef(proxy, actual)[0, 1])
    audit = {
        "schema_version": 1,
        "status": "PASS_WITH_PROVENANCE_WARNING" if not provenance["git_diff_clean"] else "PASS",
        "archive_sha256": (
            read_json(root.parent / "sanitization.json").get("source_archive_sha256")
            if (root.parent / "sanitization.json").is_file() else None
        ),
        "git_head": provenance["git_head"],
        "git_diff_clean": bool(provenance["git_diff_clean"]),
        "reports": len(reports),
        "level_audits": len(raw_rows),
        "l_gt_1_raw_points": len(lgt1),
        "negative_l_gt_1_raw_points": sum(float(row["actual_gain_bpp"]) < 0 for row in lgt1),
        "minimum_l_gt_1_raw_gain_bpp": min(float(row["actual_gain_bpp"]) for row in lgt1),
        "minimum_checkpoint_level_mean_gain_bpp": min(
            float(row["actual_gain_mean"])
            for row in summary_rows
            if int(row["levels"]) > 1
        ),
        "median_l31_checkpoint_mean_gain_bpp": endpoint_median,
        "actual_vs_quantized_gain_pearson_r": correlation,
        "actual_vs_quantized_gain_mae_bpp": float(np.mean(np.abs(actual - proxy))),
        "maximum_tail_mass": max(float(row["max_tail_mass"]) for row in raw_rows),
        "maximum_normalization_residual": max(
            float(row["normalization_residual"]) for row in raw_rows
        ),
        "unit_tests": "NOT_RUN_BY_THIS_ANALYSIS_COMMAND",
        "claim_scope": "operational conditional-entropy bridge on four frozen MNIST codecs",
        "scope_exclusions": aggregate["scientific_gate"]["scope_exclusions"],
    }
    return raw_rows, {"summary_rows": summary_rows, "audit": audit}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def create_figure(
    raw_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    output_stem: Path,
) -> None:
    configure_plot_style()
    colors = {0.0: "#000000", 200.0: "#0072B2", 5000.0: "#009E73", 50000.0: "#D55E00"}
    markers = {0.0: "o", 200.0: "s", 5000.0: "^", 50000.0: "D"}
    labels = {0.0: r"$\lambda_R=0$", 200.0: r"$\lambda_R=200$", 5000.0: r"$\lambda_R=5000$", 50000.0: r"$\lambda_R=50000$"}

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.28), constrained_layout=True)
    ax_gain, ax_memory, ax_proxy = axes

    for rate in RATES:
        rows = [row for row in summary_rows if float(row["rate_weight"]) == rate]
        x = np.array([float(row["offset_entropy_bits_per_coefficient"]) for row in rows])
        y = 1000 * np.array([float(row["actual_gain_mean"]) for row in rows])
        lower = 1000 * np.array([float(row["actual_gain_ci95_lower"]) for row in rows])
        upper = 1000 * np.array([float(row["actual_gain_ci95_upper"]) for row in rows])
        ax_gain.errorbar(
            x,
            y,
            yerr=np.vstack((y - lower, upper - y)),
            color=colors[rate],
            marker=markers[rate],
            capsize=2,
            capthick=0.7,
            label=labels[rate],
        )
    ax_gain.axhline(0, color="0.65", linewidth=0.7)
    ax_gain.set_xlabel(r"Offset-state entropy $\log_2 L$ (bits/coefficient)")
    ax_gain.set_ylabel("Actual payload gain (millibits/pixel)")
    ax_gain.legend(frameon=False, loc="upper left")

    memory_rows = [row for row in summary_rows if float(row["rate_weight"]) == 0.0]
    x_memory = [float(row["offset_entropy_bits_per_coefficient"]) for row in memory_rows]
    y_memory = [float(row["conditional_table_bytes"]) / 1024 for row in memory_rows]
    ax_memory.plot(x_memory, y_memory, color="#CC79A7", marker="o")
    for row, x, y in zip(memory_rows, x_memory, y_memory):
        ax_memory.annotate(
            f"L={int(row['levels'])}",
            (x, y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.3,
        )
    ax_memory.set_xlabel(r"Offset-state entropy $\log_2 L$ (bits/coefficient)")
    ax_memory.set_ylabel("Conditional CDF storage (KiB)")

    for rate in RATES:
        rows = [
            row for row in raw_rows
            if float(row["rate_weight"]) == rate and int(row["levels"]) > 1
        ]
        ax_proxy.scatter(
            1000 * np.array([float(row["quantized_gain_bpp"]) for row in rows]),
            1000 * np.array([float(row["actual_gain_bpp"]) for row in rows]),
            color=colors[rate],
            marker=markers[rate],
            s=14,
            alpha=0.72,
            linewidths=0.3,
            edgecolors="white",
        )
    limits = ax_proxy.get_xlim()
    lower = min(limits[0], ax_proxy.get_ylim()[0])
    upper = max(limits[1], ax_proxy.get_ylim()[1])
    ax_proxy.plot([lower, upper], [lower, upper], "--", color="0.45", linewidth=0.8)
    ax_proxy.set_xlim(lower, upper)
    ax_proxy.set_ylim(lower, upper)
    ax_proxy.set_xlabel("Quantized-CDF gain (millibits/pixel)")
    ax_proxy.set_ylabel("Actual payload gain (millibits/pixel)")

    for label, axis in zip(("(a)", "(b)", "(c)"), axes):
        axis.text(-0.18, 1.06, label, transform=axis.transAxes, fontweight="bold", va="top")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(width=0.7, length=3)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".svg"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()

    raw_rows, result = audit_and_collect(arguments.root)
    summary_rows = result["summary_rows"]
    output = arguments.output_root
    write_csv(output / "conditional_rate_v81_raw.csv", raw_rows)
    write_csv(output / "conditional_rate_v81_summary.csv", summary_rows)
    (output / "conditional_rate_v81_audit.json").write_text(
        json.dumps(result["audit"], indent=2) + "\n", encoding="utf-8"
    )
    create_figure(
        raw_rows,
        summary_rows,
        output / "fig_conditional_rate_v81",
    )
    print(json.dumps(result["audit"], indent=2))


if __name__ == "__main__":
    main()
