"""Synthetic M/O strict-advantage audit for the Bernoulli semantic codec.

Protocol ``SYNTHETIC-MO-ADVANTAGE-V1``.  Companion to the MNIST M/O receiver
audit (``MO-RECEIVER-OFFSET-V1``) and the finite-mode operational simulation
(``simulate_bernoulli_finite_mode.py``).

Theory only requires H(S|Y,V) <= H(S|Y); the MNIST testbed sits at equality
(a task-neutral regime, see RESULT_ANALYSIS_MO_RECEIVER_V9.md).  This script
demonstrates the complementary fact on the paper's own Bernoulli model: there
exists a nonempty operating region where receiver-known mode information
strictly reduces task uncertainty,

    DeltaH = H(S|Y) - H(S|Y,V) > 0.

Model M receives (Y, V); model O receives only Y.  Both quantities are exact
evaluations from ``bernoulli_mode_limited`` (mode-aware and output-only task
entropies), so no network is trained and no Monte Carlo error enters the
headline numbers.  Joint probabilities are rational functions of the exact
double inputs; an exact Fraction-based cross-check is provided for tests.

Structure:

1. an (a, b) feasibility-region scan showing where DeltaH vanishes (single
   mode and constant-only schedules: each output symbol admits only one
   posterior) and where DeltaH > 0 (mixed schedules: informative modes mixed
   with constant modes, or identity mixed with flip, leave the output-only
   receiver uncertain about which posterior applies);
2. an Rc sweep of the exact optimal K=2/3/4 schedules at the manuscript
   figure-slice parameters, reporting DeltaD = D*_K3 - D*_K4 and DeltaH at
   the optimal K=4 schedule, so the strict-gain region of the phase
   transition and the strict task-information region can be compared
   directly;
3. witness points, including the positive-correlation K=4 winner witness
   used in the cor:k4-open regression tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

from .bernoulli_mode_limited import (
    evaluate,
    hb,
    optimize_k2,
    optimize_k3,
    optimize_k4_finite_candidates,
    output_only_task_entropy,
    weights_from_ab,
)
from .simulate_bernoulli_finite_mode import (
    DEFAULT_CROSSOVER,
    DEFAULT_QX,
    DEFAULT_QY,
    DEFAULT_RATE,
    DEFAULT_TASK,
)

PROTOCOL = "SYNTHETIC-MO-ADVANTAGE-V1"
MODE_NAMES = ("identity", "flip", "zero", "one")

# Positive-correlation K=4 winner witness from test_k4_open_set_counterexample.
WITNESS = {
    "qx": 0.10,
    "qy": 0.022,
    "crossover": 0.10,
    "a_c": 0.02,
    "a_r": 0.021,
    "rc": 0.26,
}


# ---------------------------------------------------------------------------
# Exact rational cross-check
# ---------------------------------------------------------------------------


def exact_joint_sv_y(qx: float, crossover: float, weights) -> dict:
    """Exact P(V=v, S=s, Y=y) as Fractions for the four deterministic maps."""

    maps = ((0, 1), (1, 0), (0, 0), (1, 1))
    qx_f = Fraction(*qx.as_integer_ratio())
    cr_f = Fraction(*crossover.as_integer_ratio())
    joint: dict[tuple[int, int, int], Fraction] = {}
    for weight, mapping in zip(weights, maps):
        w_f = Fraction(*float(weight).as_integer_ratio())
        for x, px in ((0, 1 - qx_f), (1, qx_f)):
            for noise, pn in ((0, 1 - cr_f), (1, cr_f)):
                s = x ^ noise
                y = mapping[x]
                key = (maps.index(mapping), s, y)
                joint[key] = joint.get(key, Fraction(0)) + w_f * px * pn
    return joint


def exact_delta_h(qx: float, crossover: float, weights) -> float:
    """DeltaH from the exact rational joint; logs are evaluated in float64."""

    joint = exact_joint_sv_y(qx, crossover, weights)

    def h2(p: float) -> float:
        return hb(min(max(p, 0.0), 1.0)) if 0.0 < p < 1.0 else 0.0

    # H(S|Y,V)
    h_mode = 0.0
    for v in range(4):
        for y in (0, 1):
            p_vy = sum(float(joint.get((v, s, y), 0)) for s in (0, 1))
            if p_vy > 0.0:
                p_s1 = float(joint.get((v, 1, y), 0)) / p_vy
                h_mode += p_vy * h2(p_s1)
    # H(S|Y)
    h_out = 0.0
    for y in (0, 1):
        p_y = sum(float(joint.get((v, s, y), 0)) for v in range(4) for s in (0, 1))
        if p_y > 0.0:
            p_s1 = (
                sum(float(joint.get((v, 1, y), 0)) for v in range(4)) / p_y
            )
            h_out += p_y * h2(p_s1)
    return h_out - h_mode


# ---------------------------------------------------------------------------
# Scans
# ---------------------------------------------------------------------------


def delta_h_float(qx: float, crossover: float, weights) -> float:
    h_out = output_only_task_entropy(qx, crossover, np.asarray(weights))
    a = float(weights[0] + weights[1])
    q_s = qx + crossover - 2.0 * qx * crossover
    h_mode = a * hb(crossover) + (1.0 - a) * hb(q_s)
    return h_out - h_mode


def scan_ab_region(qx: float, qy: float, crossover: float, steps: int):
    """DeltaH over the feasible (a, b) region (weights >= 0, sum 1)."""

    rows = []
    a_values = np.linspace(0.0, 1.0, steps + 1)
    for a in a_values:
        for b in np.linspace(0.0, float(a), steps + 1):
            weights = weights_from_ab(qx, qy, float(a), float(b))
            if np.min(weights) < -1e-12:
                continue
            metric = evaluate(qx, qy, crossover, float(a), float(b))
            rows.append(
                {
                    "a": float(a),
                    "b": float(b),
                    "delta_h": metric.task_entropy_output_only
                    - metric.task_entropy_mode_aware,
                    "h_mode_aware": metric.task_entropy_mode_aware,
                    "h_output_only": metric.task_entropy_output_only,
                    "mode_entropy": metric.common_randomness,
                    "support": metric.support,
                }
            )
    return rows


def sweep_rc_schedules(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task: float,
    rc_min: float,
    rc_max: float,
    rc_step: float,
):
    """Optimal K=2/3/4 schedules versus the common-randomness budget Rc."""

    rows = []
    rc_values = np.round(np.arange(rc_min, rc_max + 1e-12, rc_step), 6)
    optimizers = {2: optimize_k2, 3: optimize_k3, 4: optimize_k4_finite_candidates}
    for rc in rc_values:
        best = {}
        for k, optimizer in optimizers.items():
            candidate = optimizer(qx, qy, crossover, rate, task, float(rc))
            if candidate is None:
                continue
            metric = candidate.metric
            best[k] = candidate
            rows.append(
                {
                    "rc": float(rc),
                    "k": k,
                    "candidate": candidate.name,
                    "weights": [float(w) for w in metric.weights],
                    "distortion": metric.distortion,
                    "payload_rate": metric.payload_rate,
                    "mode_entropy": metric.common_randomness,
                    "h_mode_aware": metric.task_entropy_mode_aware,
                    "h_output_only": metric.task_entropy_output_only,
                    "delta_h": metric.task_entropy_output_only
                    - metric.task_entropy_mode_aware,
                    "support": metric.support,
                }
            )
        if 3 in best and 4 in best:
            rows[-1]["delta_d_k3_minus_k4"] = (
                best[3].metric.distortion - best[4].metric.distortion
            )
            rows[-2]["delta_d_k3_minus_k4"] = (
                best[3].metric.distortion - best[4].metric.distortion
            )
    return rows


def witness_report() -> dict:
    """DeltaH/DeltaD at the cor:k4-open positive-correlation witness.

    Budgets follow test_k4_open_set_counterexample exactly: the task budget
    corresponds to a_C and the rate budget to a_R.
    """

    q_s = WITNESS["qx"] + WITNESS["crossover"] - 2.0 * WITNESS["qx"] * WITNESS["crossover"]
    h_s = hb(q_s)
    task = h_s - WITNESS["a_c"] * (h_s - hb(WITNESS["crossover"]))
    rate = WITNESS["a_r"] * hb(WITNESS["qx"])
    out = {}
    for k, optimizer in (
        (2, optimize_k2),
        (3, optimize_k3),
        (4, optimize_k4_finite_candidates),
    ):
        candidate = optimizer(
            WITNESS["qx"],
            WITNESS["qy"],
            WITNESS["crossover"],
            rate,
            task,
            WITNESS["rc"],
        )
        if candidate is None:
            continue
        metric = candidate.metric
        out[f"K{k}"] = {
            "candidate": candidate.name,
            "weights": [float(w) for w in metric.weights],
            "distortion": metric.distortion,
            "delta_h": metric.task_entropy_output_only
            - metric.task_entropy_mode_aware,
            "delta_h_exact_fraction": exact_delta_h(
                WITNESS["qx"], WITNESS["crossover"], metric.weights
            ),
        }
    if "K3" in out and "K4" in out:
        out["delta_d_k3_minus_k4"] = (
            out["K3"]["distortion"] - out["K4"]["distortion"]
        )
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    release_root = Path(__file__).resolve().parents[1]
    if not (release_root / ".git").exists():
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=release_root,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qx", type=float, default=DEFAULT_QX)
    parser.add_argument("--qy", type=float, default=DEFAULT_QY)
    parser.add_argument("--crossover", type=float, default=DEFAULT_CROSSOVER)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE)
    parser.add_argument("--task", type=float, default=DEFAULT_TASK)
    parser.add_argument("--rc-min", type=float, default=0.0)
    parser.add_argument("--rc-max", type=float, default=2.0)
    parser.add_argument("--rc-step", type=float, default=0.025)
    parser.add_argument("--ab-steps", type=int, default=120)
    parser.add_argument("--output-dir", default="outputs/bernoulli_mo_advantage")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[scan] (a,b) feasibility region ...")
    ab_rows = scan_ab_region(args.qx, args.qy, args.crossover, args.ab_steps)

    print("[sweep] optimal schedules vs Rc ...")
    rc_rows = sweep_rc_schedules(
        args.qx,
        args.qy,
        args.crossover,
        args.rate,
        args.task,
        args.rc_min,
        args.rc_max,
        args.rc_step,
    )
    for row in rc_rows:
        if row["k"] == 4:
            print(
                f"Rc={row['rc']:.3f} K=4 {row['candidate']}: "
                f"D={row['distortion']:.6f} "
                f"DeltaD={row.get('delta_d_k3_minus_k4', float('nan')):+.6f} "
                f"DeltaH={row['delta_h']:+.6f} "
                f"H(V)={row['mode_entropy']:.4f}"
            )

    print("[witness] cor:k4-open positive-correlation witness ...")
    witness = witness_report()

    k4_rows = [r for r in rc_rows if r["k"] == 4]
    strict_d = [r["rc"] for r in k4_rows if r.get("delta_d_k3_minus_k4", 0.0) > 1e-9]
    strict_h = [r["rc"] for r in k4_rows if r["delta_h"] > 1e-9]
    summary = {
        "protocol": PROTOCOL,
        "git_commit": _git_commit(),
        "parameters": {
            "qx": args.qx,
            "qy": args.qy,
            "crossover": args.crossover,
            "rate": args.rate,
            "task": args.task,
            "rc_grid": [args.rc_min, args.rc_max, args.rc_step],
            "ab_steps": args.ab_steps,
        },
        "ab_region": {
            "points": len(ab_rows),
            "max_delta_h": max(r["delta_h"] for r in ab_rows),
            "strict_points": sum(1 for r in ab_rows if r["delta_h"] > 1e-9),
            "min_delta_h": min(r["delta_h"] for r in ab_rows),
        },
        "rc_sweep": {
            "points_k4": len(k4_rows),
            "strict_delta_d_rc": [min(strict_d), max(strict_d)] if strict_d else None,
            "strict_delta_h_rc": [min(strict_h), max(strict_h)] if strict_h else None,
            "max_delta_h": max((r["delta_h"] for r in k4_rows), default=None),
            "max_delta_d": max(
                (r.get("delta_d_k3_minus_k4", 0.0) for r in k4_rows), default=None
            ),
        },
        "witness": witness,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "ab_region.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ab_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ab_rows)
    with (output_dir / "rc_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [k for k in rc_rows[0].keys() if k != "weights"]
        if any("delta_d_k3_minus_k4" in r for r in rc_rows):
            fieldnames.append("delta_d_k3_minus_k4")
        fieldnames += ["w_identity", "w_flip", "w_zero", "w_one"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rc_rows:
            flat = {k: v for k, v in row.items() if k != "weights"}
            flat.setdefault("delta_d_k3_minus_k4", "")
            for name, w in zip(MODE_NAMES, row["weights"]):
                flat[f"w_{name}"] = w
            writer.writerow(flat)

    print(json.dumps(summary["ab_region"], indent=2))
    print(json.dumps(summary["rc_sweep"], indent=2))
    print(f"[COMPLETE] {output_dir}")


if __name__ == "__main__":
    main()
