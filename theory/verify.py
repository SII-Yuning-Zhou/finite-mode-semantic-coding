"""Fast CPU reproduction of the paper's numerical certificates and controls.

Run from the release root with ``python -m theory.verify``. This does not run
the full Monte Carlo experiment, train neural networks, or replace analytic
proofs. It recomputes interval certificates and checks the complete archived
point-level simulation table without selecting favorable observations.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path

import numpy as np

from .bernoulli_mode_limited import (
    entropy, evaluate, optimize_k3, optimize_k4_finite_candidates,
)
from .general_finite_alphabet_vertex_certificate import (
    build_coefficients, enumerate_vertices, interval_bounds, interval_string, schedule_entropy_iv,
)
from . import k4_strict_open_set_certificate
from .simulate_bernoulli_finite_mode import (
    KnuthYaoSampler, knuth_yao_expected_bits, simulate_channel,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "theory"


def run_checks() -> dict:
    stdout = StringIO()
    with redirect_stdout(stdout):
        k4_strict_open_set_certificate.main()

    q, y, epsilon, rate, task, randomness = .18, .10, .235, .082, .90, .78
    k3 = optimize_k3(q, y, epsilon, rate, task, randomness)
    k4 = optimize_k4_finite_candidates(q, y, epsilon, rate, task, randomness)
    assert k3 is not None and k4 is not None
    assert k4.metric.support == 4
    assert k3.metric.distortion - k4.metric.distortion > .01895
    assert k4.metric.payload_rate <= rate + 1e-12
    assert k4.metric.task_entropy_mode_aware <= task + 1e-12
    assert k4.metric.common_randomness <= randomness + 1e-12

    coefficients = build_coefficients()
    vertices, counts = enumerate_vertices(coefficients)
    assert len(coefficients) == 8 and counts["total_vertex_systems"] == 210
    assert counts["ambiguous"] == 0 and len(vertices) == 12
    best_k4 = min(vertices, key=lambda x: x.distortion)
    best_k3 = min((x for x in vertices if x.support_size <= 3), key=lambda x: x.distortion)
    gap_bounds = interval_bounds(best_k3.distortion_iv - best_k4.distortion_iv)
    assert gap_bounds[0] > .018
    assert interval_bounds(schedule_entropy_iv(best_k4.weights_iv))[1] < 1.9

    with (DATA / "bernoulli_sim/bernoulli_finite_mode_sim_v1.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    archived_summary = json.loads((DATA / "bernoulli_sim/bernoulli_finite_mode_sim_v1_summary.json").read_text(encoding="utf-8"))
    assert len(rows) == archived_summary["points_simulated"] == 90
    assert {int(row["K"]) for row in rows} == {2, 3, 4}
    assert all(sum(int(row["K"]) == k for row in rows) == 30 for k in (2, 3, 4))
    assert all(row["feasible"] == "1" for row in rows)
    assert max(float(row["max_abs_dev"]) for row in rows) < .003
    # Archived probabilities are rounded to ten decimal places in the CSV.
    max_metric_difference = 0.0
    for row in rows:
        weights = np.array([float(row["w_"+name]) for name in ("identity", "flip", "zero", "one")])
        metrics = evaluate(.18, .10, .235, float(weights[0]+weights[1]), float(weights[1]))
        for attr, field in (("distortion", "ana_distortion"), ("payload_rate", "ana_payload_rate"),
                            ("task_entropy_mode_aware", "ana_task_entropy_mode_aware"),
                            ("task_entropy_output_only", "ana_task_entropy_output_only")):
            delta = abs(getattr(metrics, attr)-float(row[field]))
            max_metric_difference = max(max_metric_difference, delta)
            assert delta < 2e-9, (row["Rc"], row["K"], field, delta)

    dyadic = np.array([.5, .25, .125, .125])
    expected_bits = knuth_yao_expected_bits(dyadic)
    assert abs(expected_bits-1.75) < 1e-12
    modes, costs = KnuthYaoSampler(dyadic).draw(20_000, seed=12345)
    assert np.max(np.abs(np.bincount(modes, minlength=4)/len(modes)-dyadic)) < .02
    assert abs(float(costs.mean())-expected_bits) < .03
    reference = evaluate(.18, .10, .235, .12, .02)
    smoke = simulate_channel(.18, .235, np.array(reference.weights), n=10_000, seed=12345)
    assert abs(smoke.distortion-reference.distortion) < .03
    assert abs(smoke.target_one_mass-.10) < .02
    return {
        "status": "PASS",
        "scope": "CPU numerical reproduction; no neural training; no full Monte Carlo rerun",
        "strict_bernoulli_interval_certificate": "PASS",
        "strict_bernoulli_parameters": {"q": q, "y": y, "epsilon": epsilon, "R": rate, "C": task, "Rc": randomness},
        "strict_bernoulli_k3": asdict(k3.metric),
        "strict_bernoulli_k4": asdict(k4.metric),
        "nonbernoulli_vertex_system_counts": counts,
        "nonbernoulli_k3_distortion": float(best_k3.distortion),
        "nonbernoulli_k4_distortion": float(best_k4.distortion),
        "nonbernoulli_gap_interval": interval_string(best_k3.distortion_iv-best_k4.distortion_iv),
        "nonbernoulli_gap_display_bounds_binary64": list(gap_bounds),
        "archived_simulation_points_checked": len(rows),
        "archived_analytical_recomputation_max_abs_delta": max_metric_difference,
        "fresh_bit_sampler_expected_bits": expected_bits,
        "fresh_bit_sampler_smoke_draws": len(modes),
        "fresh_bit_sampler_smoke_measured_bits": float(costs.mean()),
        "smoke_channel_symbols": 10_000,
        "smoke_is_not_the_archived_paper_experiment": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/theory_verification.json")
    args = parser.parse_args()
    report = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
