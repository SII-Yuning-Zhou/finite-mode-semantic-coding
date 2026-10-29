"""Audit the output-only Bernoulli theory in correlation coordinates.

The ICLR model constrains H(S|Y), whereas its Bernoulli appendix evaluates
H(S|Y,U).  This file works with the literal output-only constraint.  It uses

    r = a - 2 b,

where a is the total identity/flip mass and b is the flip mass.  The variable
r alone determines the X-Y coupling, Hamming distortion, and H(S|Y).  The
remaining variable b controls payload and mode-mixture entropy.

The script checks the symbolic reduction, verifies the closed-form
unlimited-randomness DRC/RDC results against grids, and searches for a strict
K=4 versus K=3 witness under a limited common-randomness budget.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import sympy as sp


TOL = 1e-11


def hb(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def entropy(weights: tuple[float, ...]) -> float:
    return -sum(w * math.log2(w) for w in weights if w > 0.0)


def weights_from_rb(q: float, y: float, r: float, b: float) -> tuple[float, ...]:
    """Return identity, flip, constant-zero, constant-one mode weights."""

    return (
        r + b,
        b,
        1.0 - y - (1.0 - q) * r - b,
        y - q * r - b,
    )


def r_bounds(q: float, y: float) -> tuple[float, float]:
    """Exact Frechet bounds expressed in the normalized correlation r."""

    Q = q * (1.0 - q)
    t_min = max(0.0, q + y - 1.0)
    t_max = min(q, y)
    return (t_min - q * y) / Q, (t_max - q * y) / Q


def b_bounds(q: float, y: float, r: float) -> tuple[float, float] | None:
    lower = max(0.0, -r)
    upper = min(1.0 - y - (1.0 - q) * r, y - q * r)
    if lower > upper + TOL:
        return None
    return lower, upper


def joint_one_probability(q: float, y: float, r: float) -> float:
    return q * y + q * (1.0 - q) * r


def distortion_from_r(q: float, y: float, r: float) -> float:
    return q + y - 2.0 * joint_one_probability(q, y, r)


def output_task_entropy(q: float, y: float, error: float, r: float) -> float:
    """Exact H(S|Y) for S=X xor N and an exact Bernoulli target Y."""

    t = joint_one_probability(q, y, r)
    result = 0.0
    if y > 0.0:
        p_s1_y1 = error + (1.0 - 2.0 * error) * t / y
        result += y * hb(p_s1_y1)
    if y < 1.0:
        p_s1_y0 = error + (1.0 - 2.0 * error) * (q - t) / (1.0 - y)
        result += (1.0 - y) * hb(p_s1_y0)
    return result


def output_task_entropy_enumerated(
    q: float, y: float, error: float, r: float, b: float
) -> float:
    maps = ((0, 1), (1, 0), (0, 0), (1, 1))
    joint = [[0.0, 0.0], [0.0, 0.0]]
    for weight, mapping in zip(weights_from_rb(q, y, r, b), maps):
        for x, px in ((0, 1.0 - q), (1, q)):
            for noise, pn in ((0, 1.0 - error), (1, error)):
                joint[x ^ noise][mapping[x]] += weight * px * pn
    result = 0.0
    for yy in (0, 1):
        py = joint[0][yy] + joint[1][yy]
        if py > 0.0:
            result += py * hb(joint[1][yy] / py)
    return result


def bisect_task_root(
    q: float, y: float, error: float, task: float, left: float, right: float
) -> float:
    """Solve H(S|Y)=task on an interval with opposite endpoint signs."""

    f_left = output_task_entropy(q, y, error, left) - task
    f_right = output_task_entropy(q, y, error, right) - task
    if abs(f_left) <= TOL:
        return left
    if abs(f_right) <= TOL:
        return right
    if f_left * f_right > 0.0:
        raise ValueError("task root is not bracketed")
    for _ in range(180):
        middle = (left + right) / 2.0
        f_middle = output_task_entropy(q, y, error, middle) - task
        if f_left * f_middle <= 0.0:
            right = middle
            f_right = f_middle
        else:
            left = middle
            f_left = f_middle
    return (left + right) / 2.0


@dataclass(frozen=True)
class Point:
    r: float
    b: float
    distortion: float
    payload: float
    randomness: float
    task: float
    support: int
    weights: tuple[float, ...]


def point(q: float, y: float, error: float, r: float, b: float) -> Point:
    weights = weights_from_rb(q, y, r, b)
    return Point(
        r=r,
        b=b,
        distortion=distortion_from_r(q, y, r),
        payload=hb(q) * (r + 2.0 * b),
        randomness=entropy(weights),
        task=output_task_entropy(q, y, error, r),
        support=sum(w > 1e-10 for w in weights),
        weights=weights,
    )


def unlimited_drc(
    q: float, y: float, error: float, rate: float, task: float
) -> Point | None:
    """Exact output-only DRC without mode or common-randomness limits."""

    lo, hi = r_bounds(q, y)
    radius = rate / hb(q)
    left = max(lo, -radius)
    right = min(hi, radius)
    if left > right + TOL:
        return None

    # Fidelity prefers the largest r.  Check the positive branch first.
    if output_task_entropy(q, y, error, right) <= task + TOL:
        r_star = right
    elif left < 0.0 and output_task_entropy(q, y, error, left) <= task + TOL:
        # On the negative branch H(S|Y) increases strictly toward r=0.
        r_star = bisect_task_root(q, y, error, task, left, min(0.0, right))
    else:
        return None
    b_star = max(0.0, -r_star)
    return point(q, y, error, r_star, b_star)


def unlimited_rdc_high_fidelity(
    q: float, y: float, error: float, distortion: float, task: float
) -> Point | None:
    """Minimum payload in the fidelity-relevant region D <= D_ind."""

    _, r_hi = r_bounds(q, y)
    Q = q * (1.0 - q)
    d_ind = q + y - 2.0 * q * y
    d_ot = abs(q - y)
    if distortion < d_ot - TOL:
        return None
    r_d = max(0.0, (d_ind - distortion) / (2.0 * Q))

    h_ind = output_task_entropy(q, y, error, 0.0)
    h_ot = output_task_entropy(q, y, error, r_hi)
    if task < h_ot - TOL:
        return None
    if task >= h_ind - TOL:
        r_c = 0.0
    else:
        r_c = bisect_task_root(q, y, error, task, 0.0, r_hi)
    r_star = max(r_d, r_c)
    if r_star > r_hi + TOL:
        return None
    return point(q, y, error, r_star, 0.0)


def minimum_seed_b(
    q: float, y: float, r: float, randomness: float, max_modes: int
) -> float | None:
    """Minimum-payload b at fixed r under H(V) and support constraints."""

    bounds = b_bounds(q, y, r)
    if bounds is None:
        return None
    left, right = bounds

    def h(value: float) -> float:
        return entropy(weights_from_rb(q, y, r, value))

    if h(left) <= randomness + TOL:
        if sum(w > 1e-10 for w in weights_from_rb(q, y, r, left)) <= max_modes:
            return left

    if max_modes <= 2:
        # Both endpoints of the fixed-r segment must be checked.  The left
        # endpoint can have support three while the right endpoint collapses
        # to a valid two-mode mixture (for example identity+flip).
        right_weights = weights_from_rb(q, y, r, right)
        if h(right) <= randomness + TOL:
            if sum(w > 1e-10 for w in right_weights) <= max_modes:
                return right
        return None

    if max_modes == 3:
        if h(right) <= randomness + TOL:
            return right
        return None

    if h(right) > randomness + TOL:
        return None

    # H(w(r,b)) is strictly concave in the interior.  With the left endpoint
    # above Rc and the right endpoint below Rc, there is one descending root.
    for _ in range(180):
        middle = (left + right) / 2.0
        if h(middle) > randomness:
            left = middle
        else:
            right = middle
    return (left + right) / 2.0


def finite_drc_numeric(
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
    lo, hi = r_bounds(q, y)
    search_lo, search_hi = lo, hi
    best: Point | None = None
    best_r: float | None = None
    for refinement in range(refinements + 1):
        count = grid if refinement == 0 else 601
        step = (search_hi - search_lo) / max(count - 1, 1)
        for index in range(count):
            r = search_lo + index * step
            if output_task_entropy(q, y, error, r) > task + 2e-10:
                continue
            b = minimum_seed_b(q, y, r, randomness, max_modes)
            if b is None:
                continue
            candidate = point(q, y, error, r, b)
            if candidate.payload > rate + 2e-10:
                continue
            if best is None or candidate.r > best.r:
                best = candidate
                best_r = r
        if best_r is None:
            break
        search_lo = max(lo, best_r - 2.0 * step)
        search_hi = min(hi, best_r + 2.0 * step)
    return best


def symbolic_checks() -> None:
    q, y, r, b = sp.symbols("q y r b", positive=True)
    w1 = r + b
    w2 = b
    w3 = 1 - y - (1 - q) * r - b
    w4 = y - q * r - b
    weights = (w1, w2, w3, w4)
    assert sp.simplify(sum(weights) - 1) == 0
    py1 = q * w1 + (1 - q) * w2 + w4
    assert sp.simplify(py1 - y) == 0
    t = q * (w1 + w4)
    assert sp.simplify(t - (q * y + q * (1 - q) * r)) == 0
    distortion = w2 + q * w3 + (1 - q) * w4
    assert sp.simplify(distortion - (q + y - 2 * t)) == 0
    assert sp.simplify((w1 + w2) - (r + 2 * b)) == 0

    natural_entropy = -sum(w * sp.log(w) for w in weights)
    expected_derivative = sp.log(w3) + sp.log(w4) - sp.log(w1) - sp.log(w2)
    assert sp.simplify(sp.diff(natural_entropy, b) - expected_derivative) == 0


def random_formula_checks(seed: int = 20260809, trials: int = 500) -> None:
    rng = random.Random(seed)
    worst_entropy_error = 0.0
    monotonic_failures = 0
    worst_drc_grid = 0.0
    worst_rdc_grid = 0.0

    for _ in range(trials):
        q = rng.uniform(0.05, 0.48)
        y = rng.uniform(0.03, 0.48)
        error = rng.uniform(0.01, 0.45)
        lo, hi = r_bounds(q, y)
        r = rng.uniform(lo, hi)
        bounds = b_bounds(q, y, r)
        assert bounds is not None
        b = rng.uniform(*bounds)
        direct = output_task_entropy(q, y, error, r)
        enumerated = output_task_entropy_enumerated(q, y, error, r, b)
        worst_entropy_error = max(worst_entropy_error, abs(direct - enumerated))

        # H(S|Y) increases on the negative branch and decreases on the positive.
        eps = 1e-7 * max(1.0, hi - lo)
        if r < -eps:
            derivative = (
                output_task_entropy(q, y, error, r + eps) - direct
            ) / eps
            monotonic_failures += derivative < -1e-6
        elif r > eps:
            derivative = (
                output_task_entropy(q, y, error, r + min(eps, hi - r)) - direct
            ) / max(min(eps, hi - r), 1e-15)
            monotonic_failures += derivative > 1e-6

        rate = rng.uniform(0.0, hb(q) * max(abs(lo), hi))
        task = rng.uniform(
            min(output_task_entropy(q, y, error, lo), output_task_entropy(q, y, error, hi)),
            output_task_entropy(q, y, error, 0.0),
        )
        exact = unlimited_drc(q, y, error, rate, task)
        grid_best = None
        radius = rate / hb(q)
        for index in range(4001):
            rr = lo + (hi - lo) * index / 4000
            if abs(rr) > radius + 1e-12:
                continue
            if output_task_entropy(q, y, error, rr) <= task + 1e-12:
                grid_best = rr
        if exact is None:
            assert grid_best is None
        else:
            assert grid_best is not None
            worst_drc_grid = max(worst_drc_grid, abs(exact.r - grid_best))

        d_ind = q + y - 2 * q * y
        d_ot = abs(q - y)
        d0 = rng.uniform(d_ot, d_ind)
        task_hi = output_task_entropy(q, y, error, hi)
        task0 = output_task_entropy(q, y, error, 0.0)
        c0 = rng.uniform(task_hi, task0)
        exact_rdc = unlimited_rdc_high_fidelity(q, y, error, d0, c0)
        assert exact_rdc is not None
        grid_rate = math.inf
        for index in range(4001):
            rr = hi * index / 4000
            if distortion_from_r(q, y, rr) > d0 + 1e-12:
                continue
            if output_task_entropy(q, y, error, rr) > c0 + 1e-12:
                continue
            grid_rate = min(grid_rate, hb(q) * rr)
        assert math.isfinite(grid_rate)
        worst_rdc_grid = max(worst_rdc_grid, abs(exact_rdc.payload - grid_rate))

    print("Random output-only checks")
    print(f"  trials={trials}")
    print(f"  max entropy formula error={worst_entropy_error:.3e}")
    print(f"  monotonicity failures={monotonic_failures}")
    print(f"  max DRC r-grid error={worst_drc_grid:.3e}")
    print(f"  max RDC rate-grid error={worst_rdc_grid:.3e}")


def find_strict_witness(seed: int = 20260810, trials: int = 1000) -> None:
    rng = random.Random(seed)
    best_gap = 0.0
    best_data = None
    for _ in range(trials):
        q = rng.uniform(0.08, 0.42)
        y = rng.uniform(0.04, 0.46)
        error = rng.uniform(0.05, 0.35)
        randomness = rng.uniform(0.35, 1.25)
        rate = rng.uniform(0.03, 0.75 * hb(q))
        task = 1.0
        k3 = finite_drc_numeric(
            q, y, error, rate, task, randomness, 3, grid=801, refinements=1
        )
        k4 = finite_drc_numeric(
            q, y, error, rate, task, randomness, 4, grid=801, refinements=1
        )
        if k3 is None or k4 is None:
            continue
        gap = k3.distortion - k4.distortion
        if gap > best_gap + 1e-5 and k4.support == 4:
            best_gap = gap
            best_data = (q, y, error, rate, randomness, k3, k4)

    print("Strict output-only K=3 versus K=4 search")
    if best_data is None:
        print("  no witness found")
        return
    q, y, error, rate, randomness, k3, k4 = best_data
    print(
        f"  q={q:.9f}, y={y:.9f}, e={error:.9f}, "
        f"R={rate:.9f}, Rc={randomness:.9f}"
    )
    for name, result in (("K3", k3), ("K4", k4)):
        print(
            f"  {name}: D={result.distortion:.9f}, r={result.r:.9f}, "
            f"b={result.b:.9f}, payload={result.payload:.9f}, "
            f"Rc={result.randomness:.9f}, K={result.support}, "
            f"w={tuple(round(w, 9) for w in result.weights)}"
        )
    print(f"  strict distortion gap={best_gap:.9f}")


def targeted_separation_search(seed: int = 20260811, trials: int = 500) -> None:
    """Target points where an interior four-mode entropy root exists."""

    rng = random.Random(seed)
    best = None
    best_gap = 0.0
    four_only = None
    for _ in range(trials):
        q = rng.uniform(0.08, 0.42)
        y = rng.uniform(0.04, 0.46)
        error = rng.uniform(0.03, 0.35)
        lo, hi = r_bounds(q, y)
        r = rng.uniform(lo + 0.08 * (hi - lo), hi - 0.08 * (hi - lo))
        bounds = b_bounds(q, y, r)
        if bounds is None or bounds[1] - bounds[0] < 1e-4:
            continue
        left, right = bounds
        h_left = entropy(weights_from_rb(q, y, r, left))
        h_right = entropy(weights_from_rb(q, y, r, right))
        if h_right >= h_left - 1e-5:
            continue
        randomness = (h_left + h_right) / 2.0
        b4 = minimum_seed_b(q, y, r, randomness, 4)
        if b4 is None or not (left + 1e-7 < b4 < right - 1e-7):
            continue
        rate = hb(q) * (r + 2.0 * b4) + 2e-6
        if rate < 0.0:
            continue
        task = output_task_entropy(q, y, error, r) + 2e-7
        k3 = finite_drc_numeric(
            q, y, error, rate, task, randomness, 3, grid=3001, refinements=3
        )
        k4 = finite_drc_numeric(
            q, y, error, rate, task, randomness, 4, grid=3001, refinements=3
        )
        if k4 is None or k4.support != 4:
            continue
        if k3 is None:
            four_only = (q, y, error, rate, task, randomness, k4)
            break
        gap = k3.distortion - k4.distortion
        if gap > best_gap + 1e-7:
            best_gap = gap
            best = (q, y, error, rate, task, randomness, k3, k4)

    print("Targeted output-only separation search")
    if four_only is not None:
        q, y, error, rate, task, randomness, k4 = four_only
        print("  K=4 feasible while K=3 infeasible")
        print(
            f"  q={q:.9f}, y={y:.9f}, e={error:.9f}, R={rate:.9f}, "
            f"C={task:.9f}, Rc={randomness:.9f}"
        )
        print(f"  K4={k4}")
    elif best is not None:
        q, y, error, rate, task, randomness, k3, k4 = best
        print(
            f"  q={q:.9f}, y={y:.9f}, e={error:.9f}, R={rate:.9f}, "
            f"C={task:.9f}, Rc={randomness:.9f}"
        )
        print(f"  K3={k3}")
        print(f"  K4={k4}")
        print(f"  strict distortion gap={best_gap:.9f}")
    else:
        print("  no strict witness found")


def strict_witness_regression() -> None:
    """Recheck one deterministic numerical K=4 versus K=3 witness.

    This is deliberately labelled a numerical regression, not an analytic
    separation theorem.  The witness lies on the negative-correlation branch.
    """

    q = 0.326236854
    y = 0.387768557
    error = 0.328538306
    rate = 0.564340860
    task = 0.975871284
    randomness = 1.335539723
    k3 = finite_drc_numeric(
        q, y, error, rate, task, randomness, 3, grid=10001, refinements=4
    )
    k4 = finite_drc_numeric(
        q, y, error, rate, task, randomness, 4, grid=10001, refinements=4
    )
    assert k3 is not None and k4 is not None
    assert k4.support == 4
    assert k4.r > k3.r + 1e-4
    assert k3.distortion - k4.distortion > 1e-4
    print("Deterministic finite-mode numerical regression")
    print(f"  K3={k3}")
    print(f"  K4={k4}")
    print(f"  strict distortion gap={k3.distortion - k4.distortion:.9f}")


def example() -> None:
    q, y, error = 0.3, 0.25, 0.2
    lo, hi = r_bounds(q, y)
    print("ICLR Figure-3 parameters in output-only coordinates")
    print(f"  r interval=[{lo:.9f}, {hi:.9f}]")
    print(f"  H(S|Y) at independence={output_task_entropy(q,y,error,0.0):.9f}")
    print(f"  H(S|Y) at OT={output_task_entropy(q,y,error,hi):.9f}")
    result = unlimited_rdc_high_fidelity(q, y, error, 0.20, 0.90)
    print(f"  high-fidelity RDC example={result}")


def main() -> None:
    symbolic_checks()
    print("Symbolic r-b reduction: PASS")
    example()
    random_formula_checks()
    strict_witness_regression()
    find_strict_witness()
    targeted_separation_search()


if __name__ == "__main__":
    main()
