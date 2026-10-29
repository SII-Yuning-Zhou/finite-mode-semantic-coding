"""Operational finite-mode simulation for the Bernoulli semantic codec.

This script closes the gap between the analytical Bernoulli optimizers and an
operational experiment: it *runs* the finite-mode codec on an iid Bernoulli
source.  For each common-randomness budget Rc and mode budget K in {2,3,4},
the exact finite-candidate optimizer of ``bernoulli_mode_limited`` supplies
the optimal mode schedule w*.  The simulation then

1. draws an iid source X ~ Ber(qx) and task variable S = X xor Ber(eps);
2. draws the shared mode V ~ w* and emits Y = f_V(X) over the four
   deterministic maps (identity, flip, constant zero, constant one);
3. measures empirical distortion, empirical conditional payload H(Y|V),
   empirical mode-aware/output-only task uncertainty, and target mismatch;
4. measures the fresh fair-bit consumption of an exact Knuth--Yao sampler
   for w*, separately from the channel simulation.

The script is an operational check of the theorems, not a proof.  All claims
are statistical: each operating point is replicated with frozen PCG64 seeds
and reported as a mean with a 95% t-interval across replications.

Output layers are deliberately distinct:

- channel simulation draws V with a vectorized uniform construction (Tier 1);
- fresh-bit accounting uses the exact Knuth--Yao DDG-tree sampler on a
  separate draw batch (Tier 2), so bit costs are exact rather than inferred
  from the entropy formula;
- analytical reference values come from ``evaluate`` at the same weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .bernoulli_mode_limited import (
    Metrics,
    entropy,
    evaluate,
    hb,
    optimize_k2,
    optimize_k3,
    optimize_k4_finite_candidates,
)

MODE_NAMES = ("identity", "flip", "zero", "one")

# Figure-slice parameters, identical to fig_resource_curves_v1_manifest.json.
DEFAULT_QX = 0.18
DEFAULT_QY = 0.10
DEFAULT_CROSSOVER = 0.235
DEFAULT_RATE = 0.11
DEFAULT_TASK = 0.90


# ---------------------------------------------------------------------------
# Exact fair-bit sampling (Knuth--Yao DDG tree for arbitrary probabilities)
#
# The input weights are IEEE-754 doubles.  Each double is an exact dyadic
# rational; writing all of them over a common denominator 2^E gives integer
# numerators n_i, and the sampler realizes the *exact* rational distribution
# p_i = n_i / sum_j n_j.  Its binary digit streams are produced by exact
# integer remainder arithmetic, so the DDG tree is complete in the limit and
# the random walk terminates with probability one: no truncation, no restart,
# and no approximation of the target law.  Numerical reporting truncates the
# expected-bit series at a depth cap with a proved error bound below
# 8 * 2^-cap.
# ---------------------------------------------------------------------------


class _DDGTree:
    """Lazily extended Knuth--Yao DDG tree with exact integer digit streams."""

    def __init__(self, probs: np.ndarray) -> None:
        probs = np.asarray(probs, dtype=np.float64)
        if np.any(probs < 0.0) or probs.sum() <= 0.0:
            raise ValueError(
                "probabilities must be nonnegative with positive mass"
            )
        ratios = [float(p).as_integer_ratio() for p in probs]
        max_exp = max(den.bit_length() - 1 for _num, den in ratios)
        numerators = [
            num << (max_exp - (den.bit_length() - 1)) for num, den in ratios
        ]
        self._total = sum(numerators)
        self._remainders = list(numerators)
        self.degenerate = sum(n > 0 for n in numerators) <= 1
        self.leaves: dict[tuple[int, int], int] = {}
        self.mass_by_depth: dict[int, float] = {}
        self._frontier: list[int] = [0]
        self._depth = 0
        self.complete = self.degenerate

    def _step_digits(self) -> list[int]:
        """Advance every outcome's binary expansion by one digit, exactly."""
        bits: list[int] = []
        new_remainders: list[int] = []
        for rem in self._remainders:
            doubled = 2 * rem
            if doubled >= self._total:
                bits.append(1)
                new_remainders.append(doubled - self._total)
            else:
                bits.append(0)
                new_remainders.append(doubled)
        self._remainders = new_remainders
        return bits

    def ensure_depth(self, depth: int) -> None:
        """Extend the tree to ``depth`` levels (or to completeness)."""
        while self._depth < depth and not self.complete:
            self._depth += 1
            modes = [i for i, bit in enumerate(self._step_digits()) if bit]
            children: list[int] = []
            for code in self._frontier:
                children.append(2 * code)
                children.append(2 * code + 1)
            children.sort()
            if len(modes) > len(children):
                raise RuntimeError("DDG frontier invariant violated.")
            for child, mode in zip(children, modes):
                self.leaves[(self._depth, child)] = mode
            self.mass_by_depth[self._depth] = len(modes) * 2.0**-self._depth
            self._frontier = children[len(modes):]
            if not self._frontier:
                self.complete = True

    def tail(self, depth: int) -> float:
        """P(walk survives ``depth`` fair bits without a leaf)."""
        self.ensure_depth(depth)
        covered = sum(
            mass for level, mass in self.mass_by_depth.items() if level <= depth
        )
        return 1.0 - covered


def _binary_leaves(probs: np.ndarray, max_depth: int) -> list[tuple[int, int, int]]:
    """Return (depth, code, mode) leaves of the DDG tree up to ``max_depth``."""

    tree = _DDGTree(probs)
    tree.ensure_depth(max_depth)
    return sorted(
        (depth, code, mode) for (depth, code), mode in tree.leaves.items()
    )


def knuth_yao_expected_bits(probs: np.ndarray, max_depth: int = 60) -> float:
    """Expected fresh fair bits per draw for the exact DDG-tree sampler.

    The expectation is sum_{j>=0} tail_j, computed exactly up to
    ``max_depth``; the omitted tail is bounded by 8 * 2^-max_depth because
    each outcome's truncation remainder is below 2^-j at depth j.  The true
    value lies in [H(p), H(p) + 2).  A degenerate distribution costs zero
    bits.
    """

    tree = _DDGTree(probs)
    if tree.degenerate:
        return 0.0
    tree.ensure_depth(max_depth)
    expected = 0.0
    for j in range(max_depth):
        expected += tree.tail(j)
    return expected


class KnuthYaoSampler:
    """Vectorized exact DDG-tree sampler over fresh fair bits.

    Every active draw consumes one fresh fair bit per depth from a PCG64 bit
    stream until it hits a leaf.  The walk is unbounded (Python integers keep
    exact codes beyond 64 bits), so the empirical distribution is exactly the
    rational target law.  Per-draw bit costs are returned so that fresh-bit
    consumption is measured, not assumed.
    """

    def __init__(self, probs: np.ndarray, batch_depth: int = 60) -> None:
        self._tree = _DDGTree(probs)
        self.probs = np.asarray(probs, dtype=np.float64)
        self.degenerate = self._tree.degenerate
        self.batch_depth = batch_depth

    def _leaf_maps(self) -> list[dict[int, int]]:
        self._tree.ensure_depth(self.batch_depth)
        maps: list[dict[int, int]] = [
            {} for _ in range(self.batch_depth + 1)
        ]
        for (depth, code), mode in self._tree.leaves.items():
            if depth <= self.batch_depth:
                maps[depth][code] = mode
        return maps

    def draw(
        self, count: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (modes, per-draw bit costs), exactly distributed."""
        if count < 0:
            raise ValueError("count must be nonnegative.")
        if self.degenerate:
            mode = int(np.flatnonzero(self.probs > 0.0)[0])
            return (
                np.full(count, mode, dtype=np.int64),
                np.zeros(count, dtype=np.int64),
            )
        generator = np.random.Generator(np.random.PCG64(seed))
        leaf_maps = self._leaf_maps()
        modes = np.empty(count, dtype=np.int64)
        costs = np.zeros(count, dtype=np.int64)
        active = np.arange(count)
        codes = np.zeros(count, dtype=np.int64)
        for depth in range(1, self.batch_depth + 1):
            if not active.size:
                break
            bits = generator.integers(0, 2, size=active.size, dtype=np.int64)
            codes = (codes << 1) | bits
            costs[active] += 1
            leaf_map = leaf_maps[depth]
            if not leaf_map:
                continue
            hit_mode = np.full(active.size, -1, dtype=np.int64)
            for code, mode in leaf_map.items():
                hit_mode[codes == code] = mode
            done = hit_mode >= 0
            if np.any(done):
                done_index = active[done]
                modes[done_index] = hit_mode[done]
                keep = ~done
                active = active[keep]
                codes = codes[keep]
        # Deep survivors (probability below 4*2^-batch_depth each) finish
        # with an exact unbounded walk; costs continue accumulating.
        for index, code in zip(active.tolist(), codes.tolist()):
            mode, extra = self._walk_one_from(generator, code, self.batch_depth)
            modes[index] = mode
            costs[index] += extra
        return modes, costs

    def _walk_one_from(
        self, generator: np.random.Generator, code: int, depth: int
    ) -> tuple[int, int]:
        """Continue an exact walk from a given (code, depth) prefix."""
        extra = 0
        while True:
            depth += 1
            extra += 1
            self._tree.ensure_depth(depth)
            code = (code << 1) | int(
                generator.integers(0, 2, size=1, dtype=np.int64)[0]
            )
            mode = self._tree.leaves.get((depth, code))
            if mode is not None:
                return mode, extra


# ---------------------------------------------------------------------------
# Plug-in estimators
# ---------------------------------------------------------------------------


def _plug_in_conditional_entropy(
    outcome: np.ndarray, conditioning: np.ndarray, n_outcomes: int, n_cond: int
) -> float:
    """Plug-in H(outcome | conditioning) in bits from a count table."""

    table = np.zeros((n_outcomes, n_cond), dtype=np.float64)
    np.add.at(table, (outcome, conditioning), 1.0)
    total = table.sum()
    if total <= 0.0:
        return 0.0
    joint = table / total
    p_cond = joint.sum(axis=0)
    result = 0.0
    for j in range(n_cond):
        if p_cond[j] > 0.0:
            result += float(p_cond[j]) * hb(float(joint[1, j] / p_cond[j]))
    return result


@dataclass(frozen=True)
class SimulatedPoint:
    distortion: float
    payload_rate: float
    task_entropy_mode_aware: float
    task_entropy_output_only: float
    target_one_mass: float
    mode_entropy: float
    sampler_bits_per_draw: float


def simulate_channel(
    qx: float,
    crossover: float,
    weights: np.ndarray,
    n: int,
    seed: int,
) -> SimulatedPoint:
    """Run the finite-mode codec on n iid source symbols."""

    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()

    generator = np.random.Generator(np.random.PCG64(seed))
    x = (generator.random(n) < qx).astype(np.int64)
    noise = (generator.random(n) < crossover).astype(np.int64)
    s = x ^ noise

    cumulative = np.cumsum(weights)
    v = np.searchsorted(
        cumulative, generator.random(n), side="right"
    ).astype(np.int64)

    y = x.copy()
    flip_mask = v == 1
    zero_mask = v == 2
    one_mask = v == 3
    y[flip_mask] = 1 - x[flip_mask]
    y[zero_mask] = 0
    y[one_mask] = 1

    joint_sv = np.zeros((2, 2, 4), dtype=np.float64)
    np.add.at(joint_sv, (s, y, v), 1.0)
    joint_sv /= n
    p_v = joint_sv.sum(axis=(0, 1))
    task_mode_aware = 0.0
    payload = 0.0
    for mode in range(4):
        if p_v[mode] <= 0.0:
            continue
        p_y_given_v = joint_sv[:, :, mode].sum(axis=0) / p_v[mode]
        payload += float(p_v[mode]) * hb(float(p_y_given_v[1]))
        for y_value in (0, 1):
            mass = float(joint_sv[:, y_value, mode].sum())
            if mass > 0.0:
                p_s = float(joint_sv[1, y_value, mode] / mass)
                task_mode_aware += mass * hb(p_s)

    joint_sy = joint_sv.sum(axis=2)
    task_output_only = 0.0
    for y_value in (0, 1):
        mass = float(joint_sy[:, y_value].sum())
        if mass > 0.0:
            task_output_only += mass * hb(float(joint_sy[1, y_value] / mass))

    counts_v = np.bincount(v, minlength=4).astype(np.float64) / n
    return SimulatedPoint(
        distortion=float(np.mean(x != y)),
        payload_rate=payload,
        task_entropy_mode_aware=float(task_mode_aware),
        task_entropy_output_only=float(task_output_only),
        target_one_mass=float(np.mean(y)),
        mode_entropy=float(entropy(counts_v)),
        sampler_bits_per_draw=0.0,
    )


# ---------------------------------------------------------------------------
# Replication statistics
# ---------------------------------------------------------------------------


def _t_interval(values: np.ndarray) -> tuple[float, float]:
    """Mean and 95% t half-width across replications."""

    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if values.size < 2:
        return mean, 0.0
    # t_{0.975, df} for df = 1..30; beyond that 1.96 is within 3%.
    t_table = (
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
        2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,
        2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052,
        2.048, 2.045, 2.042,
    )
    df = values.size - 1
    t_crit = t_table[df - 1] if df <= 30 else 1.96
    return mean, float(t_crit * values.std(ddof=1) / math.sqrt(values.size))


@dataclass(frozen=True)
class PointRecord:
    rc: float
    k: int
    candidate: str
    weights: tuple[float, ...]
    analytical: dict[str, float]
    empirical_mean: dict[str, float]
    empirical_half_width: dict[str, float]
    knuth_yao_expected_bits: float
    knuth_yao_empirical_bits: float
    feasible_empirical: bool
    max_abs_deviation: float


def solve_optimal_weights(
    k: int,
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task: float,
    rc: float,
) -> tuple[str, Metrics] | None:
    if k == 2:
        candidate = optimize_k2(qx, qy, crossover, rate, task, rc)
    elif k == 3:
        candidate = optimize_k3(qx, qy, crossover, rate, task, rc)
    else:
        candidate = optimize_k4_finite_candidates(
            qx, qy, crossover, rate, task, rc
        )
    if candidate is None:
        return None
    return candidate.name, candidate.metric


def simulate_operating_point(
    k: int,
    rc: float,
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task: float,
    n: int,
    reps: int,
    base_seed: int,
    sampler_draws: int,
    feasibility_tol: float = 3e-3,
) -> PointRecord | None:
    solved = solve_optimal_weights(k, qx, qy, crossover, rate, task, rc)
    if solved is None:
        return None
    name, metric = solved
    weights = np.asarray(metric.weights, dtype=np.float64)

    fields = (
        "distortion",
        "payload_rate",
        "task_entropy_mode_aware",
        "task_entropy_output_only",
        "target_one_mass",
        "mode_entropy",
    )
    samples: dict[str, list[float]] = {field: [] for field in fields}
    for rep in range(reps):
        point = simulate_channel(qx, crossover, weights, n, base_seed + 7919 * rep)
        for field in fields:
            samples[field].append(getattr(point, field))

    empirical_mean: dict[str, float] = {}
    empirical_half: dict[str, float] = {}
    for field in fields:
        mean, half = _t_interval(np.asarray(samples[field]))
        empirical_mean[field] = mean
        empirical_half[field] = half

    analytical = {
        "distortion": metric.distortion,
        "payload_rate": metric.payload_rate,
        "task_entropy_mode_aware": metric.task_entropy_mode_aware,
        "task_entropy_output_only": metric.task_entropy_output_only,
        "target_one_mass": qy,
        "mode_entropy": metric.common_randomness,
    }
    deviations = [
        abs(empirical_mean[field] - analytical[field]) for field in fields
    ]
    feasible_empirical = (
        empirical_mean["payload_rate"] <= rate + feasibility_tol
        and empirical_mean["task_entropy_mode_aware"] <= task + feasibility_tol
        and metric.common_randomness <= rc + 1e-9
        and abs(empirical_mean["target_one_mass"] - qy) <= feasibility_tol
    )

    expected_bits = knuth_yao_expected_bits(weights)
    sampler = KnuthYaoSampler(weights)
    _, bit_costs = sampler.draw(sampler_draws, seed=base_seed + 5_000_000)
    empirical_bits = float(np.mean(bit_costs)) if sampler_draws > 0 else 0.0

    return PointRecord(
        rc=rc,
        k=k,
        candidate=name,
        weights=tuple(float(w) for w in weights),
        analytical=analytical,
        empirical_mean=empirical_mean,
        empirical_half_width=empirical_half,
        knuth_yao_expected_bits=expected_bits,
        knuth_yao_empirical_bits=empirical_bits,
        feasible_empirical=feasible_empirical,
        max_abs_deviation=float(max(deviations)),
    )


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


def run_grid(args: argparse.Namespace) -> list[PointRecord]:
    records: list[PointRecord] = []
    rc_values = np.round(
        np.arange(args.rc_min, args.rc_max + 1e-12, args.rc_step), 6
    )
    for rc_index, rc in enumerate(rc_values):
        for k in (2, 3, 4):
            record = simulate_operating_point(
                k=k,
                rc=float(rc),
                qx=args.qx,
                qy=args.qy,
                crossover=args.crossover,
                rate=args.rate,
                task=args.task,
                n=args.n,
                reps=args.reps,
                base_seed=args.seed + 104_729 * rc_index + 1_009 * k,
                sampler_draws=args.sampler_draws,
            )
            if record is None:
                continue
            records.append(record)
            print(
                f"Rc={rc:.3f} K={k} {record.candidate}: "
                f"D sim={record.empirical_mean['distortion']:.6f} "
                f"(+/-{record.empirical_half_width['distortion']:.2e}) "
                f"ana={record.analytical['distortion']:.6f} "
                f"max|dev|={record.max_abs_deviation:.2e} "
                f"KY bits={record.knuth_yao_empirical_bits:.4f} "
                f"(H={record.analytical['mode_entropy']:.4f}) "
                f"feasible={record.feasible_empirical}"
            )
    return records


def write_csv(records: list[PointRecord], path: Path) -> None:
    fields = (
        "distortion",
        "payload_rate",
        "task_entropy_mode_aware",
        "task_entropy_output_only",
        "target_one_mass",
        "mode_entropy",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["Rc", "K", "candidate", "w_identity", "w_flip", "w_zero", "w_one"]
        for field in fields:
            header += [f"ana_{field}", f"sim_{field}", f"hw_{field}"]
        header += ["ky_expected_bits", "ky_empirical_bits", "feasible", "max_abs_dev"]
        writer.writerow(header)
        for record in records:
            row = [
                f"{record.rc:.6f}",
                record.k,
                record.candidate,
                *(f"{w:.10f}" for w in record.weights),
            ]
            for field in fields:
                row += [
                    f"{record.analytical[field]:.10f}",
                    f"{record.empirical_mean[field]:.10f}",
                    f"{record.empirical_half_width[field]:.3e}",
                ]
            row += [
                f"{record.knuth_yao_expected_bits:.10f}",
                f"{record.knuth_yao_empirical_bits:.10f}",
                int(record.feasible_empirical),
                f"{record.max_abs_deviation:.3e}",
            ]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qx", type=float, default=DEFAULT_QX)
    parser.add_argument("--qy", type=float, default=DEFAULT_QY)
    parser.add_argument("--crossover", type=float, default=DEFAULT_CROSSOVER)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE)
    parser.add_argument("--task", type=float, default=DEFAULT_TASK)
    parser.add_argument("--rc-min", type=float, default=0.0)
    parser.add_argument("--rc-max", type=float, default=2.0)
    parser.add_argument("--rc-step", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--sampler-draws", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "bernoulli_sim",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = run_grid(args)

    csv_path = args.output_dir / "bernoulli_finite_mode_sim_v1.csv"
    write_csv(records, csv_path)

    feasible_count = sum(record.feasible_empirical for record in records)
    max_dev = max((record.max_abs_deviation for record in records), default=0.0)
    max_half = max(
        (record.empirical_half_width["distortion"] for record in records),
        default=0.0,
    )
    ky_violations = sum(
        not (
            record.analytical["mode_entropy"] - 1e-9
            <= record.knuth_yao_expected_bits
            < record.analytical["mode_entropy"] + 2.0
        )
        for record in records
    )
    strict_gain_points = [
        (record.rc, record.analytical["distortion"])
        for record in records
        if record.k == 4
    ]
    k3_at = {
        record.rc: record.analytical["distortion"]
        for record in records
        if record.k == 3
    }
    strict_region = [
        rc
        for rc, d4 in strict_gain_points
        if rc in k3_at and k3_at[rc] - d4 > 1e-6
    ]
    summary = {
        "version": 1,
        "protocol": "BERNOULLI-FINITE-MODE-OPERATIONAL-SIM-V1",
        "parameters": {
            "qx": args.qx,
            "qy": args.qy,
            "crossover": args.crossover,
            "rate": args.rate,
            "task": args.task,
        },
        "grid": {
            "rc_min": args.rc_min,
            "rc_max": args.rc_max,
            "rc_step": args.rc_step,
        },
        "statistics": {
            "n_per_rep": args.n,
            "reps": args.reps,
            "base_seed": args.seed,
            "sampler_draws": args.sampler_draws,
        },
        "git_commit": _git_commit(),
        "points_simulated": len(records),
        "points_empirically_feasible": feasible_count,
        "max_abs_deviation_vs_analytical": max_dev,
        "max_distortion_half_width": max_half,
        "knuth_yao_bound_violations": ky_violations,
        "strict_k4_gain_rc_values": strict_region,
        "csv": csv_path.name,
    }
    json_path = args.output_dir / "bernoulli_finite_mode_sim_v1_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("Bernoulli finite-mode operational simulation summary")
    print(f"  points simulated:            {len(records)}")
    print(f"  empirically feasible:        {feasible_count}/{len(records)}")
    print(f"  max |sim - analytical|:      {max_dev:.3e}")
    print(f"  max distortion half-width:   {max_half:.3e}")
    print(f"  Knuth-Yao bound violations:  {ky_violations}")
    print(
        f"  strict K=4 gain Rc values:   "
        f"{[round(rc, 3) for rc in strict_region]}"
    )
    print(f"  CSV:     {csv_path}")
    print(f"  Summary: {json_path}")


if __name__ == "__main__":
    main()
