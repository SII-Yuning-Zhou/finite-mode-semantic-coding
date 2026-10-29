"""Root-free forward-DRC audit for output-only Bernoulli model O.

The literal output-only task constraint is H(S|Y) <= C.  In the correlation
coordinates

    r = w_I - w_F,    b = w_F,

the target law, distortion, and output-only task entropy depend on r, while
the payload is h_b(q) (r + 2 b).  Once a *forward* payload budget R is fixed,
the admissible b interval is truncated by a linear rate boundary.  Since the
mode entropy is strictly concave in b, fixed-r feasibility is decided only by
the two endpoints of this truncated interval; no entropy root is needed.

This module implements that theorem, compares it with the older implicit-root
implementation, and implements the constructive positive-correlation K=3
dominance theorem.  The subsequent exact positive outer solution lives in
``output_only_positive_finite_root.py``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import sympy as sp

from .output_only_rb_theory_audit import (
    Point,
    TOL,
    b_bounds,
    distortion_from_r,
    entropy,
    hb,
    minimum_seed_b,
    output_task_entropy,
    point,
    r_bounds,
    weights_from_rb,
)


@dataclass(frozen=True)
class FixedRInterval:
    """Rate-truncated fixed-coupling interval and its endpoint entropies."""

    r: float
    rho: float
    b_left: float
    b_target: float
    b_rate: float
    b_cap: float
    cap_boundary: str
    entropy_left: float
    entropy_cap: float


def fixed_r_interval(q: float, y: float, r: float, rate: float) -> FixedRInterval | None:
    """Return the exact b interval after target and payload constraints.

    The payload constraint is r + 2 b <= rho with rho = R / h_b(q).
    This routine is for the non-degenerate source regime 0 < q < 1.
    """

    source_entropy = hb(q)
    if source_entropy <= 0.0:
        raise ValueError("fixed_r_interval requires a non-deterministic source")
    if rate < -TOL:
        return None
    target_bounds = b_bounds(q, y, r)
    if target_bounds is None:
        return None
    b_left, b_target = target_bounds
    rho = max(0.0, rate) / source_entropy
    b_rate = (rho - r) / 2.0
    b_cap = min(b_target, b_rate)
    if b_left > b_cap + TOL:
        return None
    b_cap = max(b_left, b_cap)
    cap_boundary = "target" if b_target <= b_rate + TOL else "rate"
    return FixedRInterval(
        r=r,
        rho=rho,
        b_left=b_left,
        b_target=b_target,
        b_rate=b_rate,
        b_cap=b_cap,
        cap_boundary=cap_boundary,
        entropy_left=entropy(weights_from_rb(q, y, r, b_left)),
        entropy_cap=entropy(weights_from_rb(q, y, r, b_cap)),
    )


def fixed_r_root_free_candidate(
    q: float,
    y: float,
    error: float,
    r: float,
    rate: float,
    task: float,
    randomness: float,
    max_modes: int,
) -> Point | None:
    """Exact fixed-r forward feasibility for K in {3,4}, without entropy roots."""

    if max_modes not in (3, 4):
        raise ValueError("the root-free endpoint theorem is implemented for K=3 or K=4")
    lo, hi = r_bounds(q, y)
    if r < lo - TOL or r > hi + TOL:
        return None
    if output_task_entropy(q, y, error, r) > task + TOL:
        return None
    interval = fixed_r_interval(q, y, r, rate)
    if interval is None:
        return None

    endpoint_candidates: list[tuple[str, float]] = [("map", interval.b_left)]
    if max_modes == 4:
        endpoint_candidates.append((interval.cap_boundary, interval.b_cap))
    elif interval.b_target <= interval.b_rate + TOL:
        endpoint_candidates.append(("target", interval.b_target))

    # At fixed r all feasible endpoints have the same objective value.  Prefer
    # the lower-payload endpoint when both pass, which also makes results stable.
    endpoint_candidates.sort(key=lambda item: item[1])
    seen: list[float] = []
    for _, b in endpoint_candidates:
        if any(abs(b - old) <= 1e-13 for old in seen):
            continue
        seen.append(b)
        candidate = point(q, y, error, r, b)
        if min(candidate.weights) < -2e-10:
            continue
        if candidate.support > max_modes:
            continue
        if candidate.payload > rate + 2e-10:
            continue
        if candidate.randomness > randomness + 2e-10:
            continue
        return candidate
    return None


def same_r_k4_only_window(
    q: float,
    y: float,
    r: float,
    rate: float,
    randomness: float,
) -> bool:
    """Necessary and sufficient K=4-only window at one fixed coupling r.

    The task constraint is omitted because it depends only on r and therefore
    cannot distinguish K=3 from K=4 at the same r.
    """

    interval = fixed_r_interval(q, y, r, rate)
    if interval is None:
        return False
    strict_rate_cut = (
        interval.b_left + TOL < interval.b_rate
        and interval.b_rate + TOL < interval.b_target
    )
    return (
        strict_rate_cut
        and interval.entropy_cap <= randomness + TOL
        and interval.entropy_left > randomness + TOL
    )


def majorizes(
    dominant: tuple[float, ...], subordinate: tuple[float, ...], tol: float = 2e-10
) -> bool:
    """Return whether ``dominant`` majorizes ``subordinate``."""

    left = sorted((max(0.0, x) for x in dominant), reverse=True)
    right = sorted((max(0.0, x) for x in subordinate), reverse=True)
    if len(left) != len(right):
        return False
    if abs(sum(left) - sum(right)) > tol:
        return False
    return all(
        sum(left[:index]) >= sum(right[:index]) - tol
        for index in range(1, len(left))
    )


def positive_k3_dominator(
    q: float, y: float, error: float, r: float, b: float
) -> Point:
    """Construct a K<=3 schedule dominating any positive-r K<=4 schedule.

    The first candidate removes flip mass while keeping r fixed.  If that
    candidate does not majorize the original mode distribution, the second
    candidate removes flip mass and raises the coupling to

        r' = min(r + 2 b, r_+).

    The majorization theorem proves that the second candidate must then
    majorize the original distribution.  Consequently it uses no more payload
    or common randomness, has no larger distortion or output-only task entropy,
    and uses at most three codec modes.
    """

    if not 0.0 < q < 1.0:
        raise ValueError("positive_k3_dominator requires 0 < q < 1")
    if not 0.0 <= error <= 0.5:
        raise ValueError("the positive-branch task ordering assumes 0 <= error <= 1/2")
    if r < -TOL:
        raise ValueError("the dominance theorem applies to r >= 0")
    bounds = b_bounds(q, y, r)
    if bounds is None or b < bounds[0] - TOL or b > bounds[1] + TOL:
        raise ValueError("(r,b) is outside the exact-target weight polytope")

    original = point(q, y, error, r, b)
    same_r = point(q, y, error, r, 0.0)
    if majorizes(same_r.weights, original.weights):
        dominant = same_r
    else:
        _, r_hi = r_bounds(q, y)
        r_prime = min(r + 2.0 * b, r_hi)
        dominant = point(q, y, error, r_prime, 0.0)
        if not majorizes(dominant.weights, original.weights):
            raise AssertionError("positive-branch majorization converse failed")

    if dominant.support > 3:
        raise AssertionError("constructed dominator is not a K<=3 schedule")
    if dominant.payload > original.payload + 3e-10:
        raise AssertionError("constructed dominator uses more payload")
    if dominant.randomness > original.randomness + 3e-10:
        raise AssertionError("constructed dominator uses more common randomness")
    if dominant.distortion > original.distortion + 3e-10:
        raise AssertionError("constructed dominator has larger distortion")
    if dominant.task > original.task + 3e-10:
        raise AssertionError("constructed dominator has larger output-only task entropy")
    return dominant


def finite_drc_root_free_numeric(
    q: float,
    y: float,
    error: float,
    rate: float,
    task: float,
    randomness: float,
    max_modes: int,
    grid: int = 5001,
    refinements: int = 3,
) -> Point | None:
    """One-dimensional outer search with exact root-free fixed-r feasibility."""

    lo, hi = r_bounds(q, y)
    search_lo, search_hi = lo, hi
    best: Point | None = None
    best_r: float | None = None
    for refinement in range(refinements + 1):
        count = grid if refinement == 0 else 601
        step = (search_hi - search_lo) / max(count - 1, 1)
        for index in range(count):
            r = search_lo + index * step
            candidate = fixed_r_root_free_candidate(
                q, y, error, r, rate, task, randomness, max_modes
            )
            if candidate is not None and (best is None or candidate.r > best.r):
                best = candidate
                best_r = r
        if best_r is None:
            break
        search_lo = max(lo, best_r - 2.0 * step)
        search_hi = min(hi, best_r + 2.0 * step)
    return best


def minimum_k3_entropy_above(
    q: float, y: float, r_floor: float, rho: float
) -> tuple[float, float, float, str] | None:
    """Exact finite critical-point audit for positive-branch K=3 schedules.

    It minimizes H(V) over K<=3 schedules with r >= r_floor and
    r+2b <= rho.  On b=0 and on each target face, entropy is concave in r;
    hence only interval endpoints, face switches, and rate intersections are
    candidates.  This helper was used during the numerical phase preceding the
    constructive positive K=3 dominance theorem and is retained as a
    regression oracle.
    """

    _, r_hi = r_bounds(q, y)
    if r_floor < -TOL or r_floor > r_hi + TOL:
        return None
    candidates: list[tuple[float, float, float, str]] = []

    # Mapping endpoint b=0: payload equals r.  Entropy is concave in r.
    b0_hi = min(r_hi, rho)
    if r_floor <= b0_hi + TOL:
        for r in (r_floor, b0_hi):
            weights = weights_from_rb(q, y, r, 0.0)
            if min(weights) >= -2e-10:
                candidates.append((entropy(weights), r, 0.0, "map"))

    c = 1.0 - 2.0 * q
    critical = [r_floor, r_hi, rho]
    if abs(c) > 1e-14:
        critical.extend(
            (
                (1.0 - 2.0 * y) / c,
                (rho - 2.0 * y) / c,
                (2.0 * (1.0 - y) - rho) / c,
            )
        )

    for r in critical:
        if r < r_floor - 2e-10 or r > r_hi + 2e-10:
            continue
        bounds = b_bounds(q, y, r)
        if bounds is None:
            continue
        b = bounds[1]
        if r + 2.0 * b > rho + 2e-10:
            continue
        weights = weights_from_rb(q, y, r, b)
        if min(weights) < -2e-10:
            continue
        candidates.append((entropy(weights), r, b, "target"))

    return min(candidates) if candidates else None


def symbolic_checks() -> None:
    q, y, r, b, rho = sp.symbols("q y r b rho", real=True)
    w_i = r + b
    w_f = b
    w_0 = 1 - y - (1 - q) * r - b
    w_1 = y - q * r - b
    weights = (w_i, w_f, w_0, w_1)

    assert sp.simplify(sum(weights) - 1) == 0
    assert sp.simplify((w_i + w_f) - (r + 2 * b)) == 0
    b_rate = (rho - r) / 2
    assert sp.simplify(r + 2 * b_rate - rho) == 0

    natural_entropy = -sum(w * sp.log(w) for w in weights)
    second = sp.diff(natural_entropy, b, 2)
    expected = -(1 / w_i + 1 / w_f + 1 / w_0 + 1 / w_1)
    assert sp.simplify(second - expected) == 0

    # In the higher-r branch r' = r+2b, the two nontrivial top-two
    # majorization inequalities reduce to nonnegative multiples of b.
    alpha, beta = sp.symbols("alpha beta", nonnegative=True)
    alpha_prime = alpha - 2 * q * b
    beta_prime = beta - 2 * (1 - q) * b
    assert sp.simplify(
        b + alpha_prime - (alpha - b) - 2 * (1 - q) * b
    ) == 0
    assert sp.simplify(b + beta_prime - (beta - b) - 2 * q * b) == 0


def random_equivalence_audit(seed: int = 2026080901, trials: int = 30000) -> None:
    """Compare root-free feasibility with the older implicit-root algorithm."""

    rng = random.Random(seed)
    mismatches = 0
    for _ in range(trials):
        q = rng.uniform(0.01, 0.99)
        y = rng.uniform(0.01, 0.99)
        error = rng.uniform(0.0, 0.49)
        lo, hi = r_bounds(q, y)
        r = rng.uniform(lo, hi)
        target_bounds = b_bounds(q, y, r)
        assert target_bounds is not None
        b_left, b_target = target_bounds
        rho_min = r + 2.0 * b_left
        rho_max = r + 2.0 * b_target
        rho = rng.uniform(max(0.0, rho_min - 0.1), max(0.0, rho_max + 0.1))
        rate = rho * hb(q)
        randomness = rng.uniform(0.0, 2.0)
        task = rng.uniform(
            output_task_entropy(q, y, error, r), 1.0
        )

        for max_modes in (3, 4):
            old_b = minimum_seed_b(q, y, r, randomness, max_modes)
            old_feasible = old_b is not None and hb(q) * (r + 2.0 * old_b) <= rate + 2e-10
            new_feasible = fixed_r_root_free_candidate(
                q, y, error, r, rate, task, randomness, max_modes
            ) is not None
            mismatches += old_feasible != new_feasible

    assert mismatches == 0
    print(f"Root-based versus root-free fixed-r audit: {2 * trials} checks, 0 mismatches")


def positive_branch_dominance_audit(
    seed: int = 2026080902, trials: int = 1_000_000
) -> None:
    """Stress-test the constructive positive-branch K=3 dominance theorem."""

    rng = random.Random(seed)
    violations = 0
    smallest_entropy_margin = math.inf
    smallest_coupling_gain = math.inf
    closest = None
    for _ in range(trials):
        q = rng.uniform(0.001, 0.999)
        y = rng.uniform(0.001, 0.999)
        error = rng.uniform(0.0, 0.5)
        _, hi = r_bounds(q, y)
        r = rng.uniform(0.0, hi)
        bounds = b_bounds(q, y, r)
        assert bounds is not None
        b_target = bounds[1]
        if b_target <= 1e-10:
            continue
        b = rng.uniform(1e-10, b_target * (1.0 - 1e-10))
        original = point(q, y, error, r, b)
        try:
            k3 = positive_k3_dominator(q, y, error, r, b)
        except AssertionError:
            violations += 1
            closest = (q, y, error, r, b, original)
            break

        entropy_margin = original.randomness - k3.randomness
        coupling_gain = k3.r - original.r
        if entropy_margin < smallest_entropy_margin:
            smallest_entropy_margin = entropy_margin
            closest = (q, y, error, r, b, original, k3)
        smallest_coupling_gain = min(smallest_coupling_gain, coupling_gain)

    assert violations == 0
    print(
        "Positive-branch constructive K=3 dominance audit: "
        f"{trials} schedules, 0 violations"
    )
    print(
        "  smallest sampled entropy margin H(original)-H(K3)="
        f"{smallest_entropy_margin:.3e}"
    )
    print(f"  smallest sampled coupling gain={smallest_coupling_gain:.3e}")
    print(f"  closest sample={closest}")


def clean_same_r_window_example() -> None:
    q, y, error = 0.10, 0.33, 0.20
    r = 0.43
    rho = 0.99
    rate = rho * hb(q)
    task = 0.90
    randomness = 1.00
    k3 = fixed_r_root_free_candidate(
        q, y, error, r, rate, task, randomness, 3
    )
    k4 = fixed_r_root_free_candidate(
        q, y, error, r, rate, task, randomness, 4
    )
    assert k3 is None
    assert k4 is not None and k4.support == 4
    assert same_r_k4_only_window(q, y, r, rate, randomness)
    print("Clean positive-r same-coupling K=4-only window")
    print(f"  q={q}, y={y}, e={error}, r={r}, rho={rho}, R={rate:.12f}")
    print(f"  C={task}, Rc={randomness}, K3={k3}")
    print(f"  K4={k4}")
    print("  This is a fixed-r separation, not a global DRC separation.")


def negative_branch_global_regression() -> None:
    q = 0.326236854
    y = 0.387768557
    error = 0.328538306
    rate = 0.564340860
    task = 0.975871284
    randomness = 1.335539723
    k3 = finite_drc_root_free_numeric(
        q, y, error, rate, task, randomness, 3, grid=10001, refinements=4
    )
    k4 = finite_drc_root_free_numeric(
        q, y, error, rate, task, randomness, 4, grid=10001, refinements=4
    )
    assert k3 is not None and k4 is not None
    assert k4.r > k3.r + 1e-4
    assert k3.distortion - k4.distortion > 1e-4
    print("Root-free negative-branch global separation regression")
    print(f"  K3={k3}")
    print(f"  K4={k4}")
    print(f"  distortion gap={k3.distortion - k4.distortion:.9f}")


def main() -> None:
    symbolic_checks()
    print("Symbolic rate-truncated endpoint reduction: PASS")
    clean_same_r_window_example()
    random_equivalence_audit()
    positive_branch_dominance_audit()
    negative_branch_global_regression()


if __name__ == "__main__":
    main()
