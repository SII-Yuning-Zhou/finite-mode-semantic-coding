"""Build the edge-profile ledger from standalone frozen-codec measurements."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def aggregate(root: Path, output: Path) -> dict:
    reports = []
    rows = []
    seen = set()
    for path in sorted(root.glob("edge_profile_rate_*.json")):
        if path.name.endswith(".config.json"):
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("profile_protocol") != "ICLR-MNIST-EDGE-PROFILE-V1" or report.get("status") != "PASS":
            raise ValueError(f"Incomplete or incompatible profile: {path}")
        key = (report["hardware"]["device"], report["rate_weight"], report["seed"])
        if key in seen:
            raise ValueError(f"Duplicate profile: {key}")
        seen.add(key)
        reports.append({"path": str(path), "rate_weight": report["rate_weight"], "seed": report["seed"], "hardware": report["hardware"], "measurement_protocol": report["measurement_protocol"], "model_resources": report["model_resources"], "checkpoint_sha256": report["checkpoint_sha256"]})
        resources = report["model_resources"]
        for point in report["levels"]:
            for name, measured in point["paths"].items():
                sender_memory = measured["sender_memory"]
                receiver_memory = measured["receiver_memory"]
                table_bytes = point[f"{name}_table_bytes"]
                rows.append({
                    "device": key[0], "rate_weight": key[1], "seed": key[2], "levels": point["levels"], "path": name,
                    "encoder_parameters": resources["encoder_parameters"],
                    "encoder_parameter_bytes": resources["encoder_parameter_bytes"],
                    "sender_tensor_state_plus_cdf_bytes": resources["encoder_tensor_state_bytes"] + resources["entropy_tensor_state_bytes"] + table_bytes,
                    "cdf_bytes": table_bytes,
                    "encode_mean_ms": measured["encode"]["mean_ms"], "encode_p95_ms": measured["encode"]["p95_ms"],
                    "decode_mean_ms": measured["decode"]["mean_ms"], "decode_p95_ms": measured["decode"]["p95_ms"],
                    "sender_rss_increment_bytes": sender_memory["host_rss_peak_increment_sampled_bytes"],
                    "receiver_rss_increment_bytes": receiver_memory["host_rss_peak_increment_sampled_bytes"],
                    "sender_cuda_peak_bytes": sender_memory.get("cuda_allocated_peak_bytes"),
                    "sender_cuda_increment_bytes": sender_memory.get("cuda_allocated_peak_increment_bytes"),
                    "receiver_cuda_peak_bytes": receiver_memory.get("cuda_allocated_peak_bytes"),
                    "receiver_cuda_increment_bytes": receiver_memory.get("cuda_allocated_peak_increment_bytes"),
                })
    if not reports:
        raise ValueError(f"No edge-profile reports found in {root}")
    result = {"schema_version": 1, "status": "PASS", "reports": reports, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Frozen MNIST edge-endpoint resource ledger", "", "Batch one; FP32; single CPU thread; setup excluded. Sender timing includes the neural encoder, shared schedule, quantization, indexed rANS and 32-bit frame. Receiver timing includes rANS, neural decoding and output transfer. Host RSS is the loaded benchmark-process footprint; sender/receiver workload increments are recorded separately.", "", "| Device | Rate weight | L | Params (M) | Codec MACs (M) | CDF KiB | Sender state KiB | Sender RSS +KiB | Encode mean / p95 ms | Decode mean / p95 ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["path"] == "conditional" and row["levels"] in (1, 31):
            report = next(item for item in reports if item["hardware"]["device"] == row["device"] and item["rate_weight"] == row["rate_weight"] and item["seed"] == row["seed"])
            resources = report["model_resources"]
            lines.append(f"| {row['device']} | {row['rate_weight']:g} | {row['levels']} | {resources['parameters']/1e6:.3f} | {resources['codec_conv_macs']/1e6:.3f} | {row['cdf_bytes']/1024:.2f} | {row['sender_tensor_state_plus_cdf_bytes']/1024:.2f} | {row['sender_rss_increment_bytes']/1024:.2f} | {row['encode_mean_ms']:.3f} / {row['encode_p95_ms']:.3f} | {row['decode_mean_ms']:.3f} / {row['decode_p95_ms']:.3f} |")
    lines.extend([
        "", "## Paired paths at fixed L = 31", "",
        "Each pair uses the same checkpoint, input images, state indexes, and reconstructed output. Times are per-image means. The delta is conditional minus marginal; it is distinct from the change between L = 1 and L = 31 above.", "",
        "| Device | Rate weight | Seed | CDF KiB, marginal / conditional | Encode ms, marginal / conditional | Encode delta ms | Decode ms, marginal / conditional | Decode delta ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for conditional in rows:
        if conditional["path"] != "conditional" or conditional["levels"] != 31:
            continue
        marginal = next(row for row in rows if row["path"] == "marginal" and all(
            row[key] == conditional[key] for key in ("device", "rate_weight", "seed", "levels")
        ))
        encode_delta = conditional["encode_mean_ms"] - marginal["encode_mean_ms"]
        decode_delta = conditional["decode_mean_ms"] - marginal["decode_mean_ms"]
        lines.append(
            f"| {conditional['device']} | {conditional['rate_weight']:g} | {conditional['seed']} "
            f"| {marginal['cdf_bytes']/1024:.2f} / {conditional['cdf_bytes']/1024:.2f} "
            f"| {marginal['encode_mean_ms']:.3f} / {conditional['encode_mean_ms']:.3f} | {encode_delta:+.3f} "
            f"| {marginal['decode_mean_ms']:.3f} / {conditional['decode_mean_ms']:.3f} | {decode_delta:+.3f} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.root, args.output)
    print(f"[PASS] {len(report['reports'])} profiles, {len(report['rows'])} ledger rows: {args.output}")


if __name__ == "__main__":
    main()
