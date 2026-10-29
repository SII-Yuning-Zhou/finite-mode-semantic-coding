"""Verify released neural numerical evidence without PyTorch or GPU access."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
T95 = 2.7764451051977987


def main():
    data = ROOT / "data/neural"
    raw = list(csv.DictReader((data / "conditional_rate_v81_raw.csv").open(encoding="utf-8", newline="")))
    summary = list(csv.DictReader((data / "conditional_rate_v81_summary.csv").open(encoding="utf-8", newline="")))
    assert len(raw) == 100 and len(summary) == 20
    by_key = {(float(x["rate_weight"]), int(x["repetition"]), int(x["levels"])): x for x in raw}
    assert len(by_key) == 100
    groups = defaultdict(list)
    for row in raw:
        key = float(row["rate_weight"]), int(row["levels"])
        groups[key].append(row)
        assert math.isclose(float(row["marginal_actual_bpp"]) - float(row["conditional_actual_bpp"]), float(row["actual_gain_bpp"]), abs_tol=1e-12)
        assert int(row["conditional_table_bytes"]) == int(row["levels"]) * int(row["marginal_table_bytes"])
        assert float(row["actual_gain_bpp"]) >= 0
        if key[1] == 1:
            assert float(row["actual_gain_bpp"]) == 0
    assert len(groups) == 20
    for row in summary:
        key = float(row["rate_weight"]), int(row["levels"])
        rows = groups[key]
        assert len(rows) == 5 and {int(x["repetition"]) for x in rows} == set(range(5))
        values = [float(x["actual_gain_bpp"]) for x in rows]
        mean = statistics.mean(values)
        half = T95 * statistics.stdev(values) / math.sqrt(5)
        for field, expected in (("actual_gain_mean", mean), ("actual_gain_ci95_lower", mean-half), ("actual_gain_ci95_upper", mean+half)):
            assert math.isclose(float(row[field]), expected, abs_tol=1e-12), (key, field)
    reports = sorted((data / "v81_reports").glob("rate_*_rep_*.json"))
    reports = [p for p in reports if not p.name.endswith(".config.json")]
    assert len(reports) == 20
    for path in reports:
        report = json.loads(path.read_text())
        assert report["status"] == "PASS" and report["audit_protocol"] == "UQ-CONDITIONAL-RANS-ESCAPE-V2"
        assert report["samples"] == 5000 and len(report["points"]) == 5
        assert report["technical_gate"]["status"] == "PASS" and not report["technical_gate"]["failures"]
        assert all(p["same_symbols_and_reconstruction"] for p in report["points"])
        for point in report["points"]:
            assert point["status"] == "PASS" and point["bitstream_roundtrip"] == "PASS_BIT_EXACT"
            assert point["max_support_normalization_residual"] <= 1e-6
            row = by_key[(float(report["rate_weight"]), int(report["schedule_repetition"]), int(point["levels"]))]
            for report_field, csv_field in (("actual_conditioning_gain_bpp", "actual_gain_bpp"),
                                           ("conditional_actual_bpp", "conditional_actual_bpp"),
                                           ("marginal_actual_bpp", "marginal_actual_bpp"),
                                           ("mse_x", "mse_x"), ("mse_hr", "mse_hr"), ("accuracy", "accuracy")):
                assert math.isclose(float(point[report_field]), float(row[csv_field]), abs_tol=1e-12)
    endpoints = []
    for rate in (0, 200, 5000, 50000):
        rows = groups[(rate, 31)]
        marginal = statistics.mean(float(x["marginal_actual_bpp"]) for x in rows)
        conditional = statistics.mean(float(x["conditional_actual_bpp"]) for x in rows)
        gain = marginal - conditional
        endpoints.append({"lambda_R": rate, "marginal_bpp": marginal, "conditional_bpp": conditional,
                          "saved_bits_per_image": 784*gain, "saved_payload_percent": 100*gain/marginal})
    print(json.dumps({"status": "PASS", "raw_observations": 100, "paired_L1_controls": 20,
                      "nontrivial_observations": 80, "groups": 20, "repetitions_per_group": 5,
                      "table_IV_recomputed": endpoints, "torch_or_checkpoint_required": False,
                      "scope": "Data arithmetic and archived report checks; not a rerun of training, ANS, or latency measurements"}, indent=2))


if __name__ == "__main__":
    main()
