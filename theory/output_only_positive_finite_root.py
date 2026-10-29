"""Finite-root positive-branch forward DRC for Bernoulli model O.

For K in {3, 4}, the positive-branch majorization converse reduces every
feasible schedule to the single map face b=0.  The remaining optimization is

    maximize r
    subject to 0 <= r <= min(r_+, R / h_b(q)),
               G(r) = H(S|Y) <= C,
               Psi(r) = H(r, beta(r), alpha(r)) <= R_c.

The task constraint gives a left endpoint r_C.  Strict concavity of Psi means
that its sublevel set on [r_C, r_U] is decided by the two endpoint entropies.
The optimizer is either r_U or the unique level root Psi(r)=R_c separating a
left sublevel interval from a right superlevel interval.  Thus the exact
positive DRC needs at most one task root and one common-randomness root.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import sympy as sp

from .output_only_rb_theory_audit import (
    Point,
    TOL,
    hb,
    output_task_entropy,
    point,
    r_bounds,
)
from .output_only_root_free_drc import fixed_r_root_free_candidate


@dataclass(frozen=True)
class PositiveFiniteRootResult:
    """Exact optimizer and the finite boundaries used to certify it."""

    optimizer: Point
    r_task: float
    r_upper: float
    active_boundary: str
    task_root_used: bool
    randomness_root_used: bool
    entropy_at_task: float
    entropy_at_upper: float


def map_face_probabilities(q: float, y: float, r: float) -> tuple[float, float, float]:
    """Return the nonzero-candidate weights (identity, constant-zero, constant-one)."""

    beta = 1.0 - y - (1.0 - q) * r
    alpha = y - q * r
    return r, beta, alpha


def map_face_entropy(q: float, y: float, r: float) -> float:
    """Common-randomness entropy Psi(r) on the dominating b=0 map face."""

    weights = map_face_probabilities(q, y, r)
    if min(weights) < -2e-10:
        raise ValueError("r is outside the positive exact-target interval")
    return -sum(weight * math.log2(weight) for weight in weights if weight > 0.0)


def map_face_entropy_derivative(q: float, y: float, r: float) -> float:
    """Derivative Psi'(r) at an interior map-face point."""

    identity, beta, alpha = map_face_probabilities(q, y, r)
    if min((identity, beta, alpha)) <= 0.0:
        raise ValueError("the derivative requires positive map-face weights")
    return math.log2((alpha**q) * (beta ** (1.0 - q)) / identity)


def map_face_entropy_second_derivative(q: float, y: float, r: float) -> float:
    """Strictly negative second derivative Psi''(r) in bits."""

    identity, beta, alpha = map_face_probabilities(q, y, r)
    if min((identity, beta, alpha)) <= 0.0:
        raise ValueError("the second derivative requires positive map-face weights")
    return -(
        1.0 / identity
        + (1.0 - q) ** 2 / beta
        + q**2 / alpha
    ) / math.log(2.0)


def _bisect_task_floor(
    q: float, y: float, error: float, task: float, right: float
) -> float:
    """Return the unique positive root G(r)=C bracketed by 0 and right."""

    left = 0.0
    f_left = output_task_entropy(q, y, error, left) - task
    f_right = output_task_entropy(q, y, error, right) - task
    if abs(f_left) <= 2e-13:
        return left
    if abs(f_right) <= 2e-13:
        return right
    if not (f_left > 0.0 and f_right < 0.0):
        raise ValueError("the decreasing positive-branch task root is not bracketed")
    for _ in range(180):
        middle = (left + right) / 2.0
        f_middle = output_task_entropy(q, y, error, middle) - task
        if f_middle > 0.0:
            left = middle
        else:
            right = middle
    return (left + right) / 2.0


def _bisect_randomness_boundary(
    q: float, y: float, randomness: float, left: float, right: float
) -> float:
    """Return the unique root Psi(r)=R_c from sublevel-left to superlevel-right."""

    f_left = map_face_entropy(q, y, left) - randomness
    f_right = map_face_entropy(q, y, right) - randomness
    if abs(f_left) <= 2e-13:
        return left
    if abs(f_right) <= 2e-13:
        return right
    if not (f_left < 0.0 and f_right > 0.0):
        raise ValueError("the positive common-randomness root is not bracketed")
    # Strict concavity and the endpoint signs make this the only level root on
    # [left, right], even if Psi is not monotone on its full target interval.
    for _ in range(180):
        middle = (left + right) / 2.0
        f_middle = map_face_entropy(q, y, middle) - randomness
        if f_middle <= 0.0:
            left = middle
        else:
            right = middle
    return (left + right) / 2.0


def positive_finite_root_drc(
    q: float,
    y: float,
    error: float,
    rate: float,
    task: float,
    randomness: float,
    max_modes: int = 3,
) -> PositiveFiniteRootResult | None:
    """Solve the exact positive-branch forward DRC for K=3 or K=4.

    ``None`` means that no r>=0 schedule meets all budgets.  Degenerate source
    and target boundaries are deliberately handled by the separate P0014
    routines; this theorem assumes 0<q,y<1 and 0<=error<=1/2.
    """

    if max_modes not in (3, 4):
        raise ValueError("the positive dominance theorem applies to K=3 or K=4")
    if not 0.0 < q < 1.0 or not 0.0 < y < 1.0:
        raise ValueError("positive_finite_root_drc requires 0<q,y<1")
    if not 0.0 <= error <= 0.5:
        raise ValueError("the task ordering requires 0<=error<=1/2")
    if rate < -TOL or randomness < -TOL:
        return None

    _, r_target = r_bounds(q, y)
    rho = max(0.0, rate) / hb(q)
    r_upper = min(r_target, rho)
    if r_upper < -TOL:
        return None
    r_upper = max(0.0, r_upper)

    task_at_zero = output_task_entropy(q, y, error, 0.0)
    task_at_upper = output_task_entropy(q, y, error, r_upper)
    if task_at_upper > task + 2e-11:
        return None
    if task_at_zero <= task + 2e-11:
        r_task = 0.0
        task_root_used = False
    else:
        r_task = _bisect_task_floor(q, y, error, task, r_upper)
        task_root_used = True

    entropy_at_task = map_face_entropy(q, y, r_task)
    entropy_at_upper = map_face_entropy(q, y, r_upper)

    if entropy_at_upper <= randomness + 2e-11:
        r_star = r_upper
        randomness_root_used = False
        if abs(r_upper - r_target) <= 2e-11 and abs(r_upper - rho) <= 2e-11:
            active_boundary = "target-rate"
        elif r_target <= rho + 2e-11:
            active_boundary = "target"
        else:
            active_boundary = "rate"
    elif entropy_at_task <= randomness + 2e-11:
        r_star = _bisect_randomness_boundary(
            q, y, randomness, r_task, r_upper
        )
        randomness_root_used = True
        active_boundary = "common-randomness"
    else:
        # A concave scalar function attains its minimum on a closed interval at
        # an endpoint.  Both endpoints above R_c therefore imply infeasibility.
        return None

    optimizer = point(q, y, error, r_star, 0.0)
    if optimizer.support > 3:
        raise AssertionError("the map-face optimizer exceeds three modes")
    if optimizer.payload > rate + 5e-10:
        raise AssertionError("the finite-root optimizer violates the payload budget")
    if optimizer.task > task + 5e-10:
        raise AssertionError("the finite-root optimizer violates the task budget")
    if optimizer.randomness > randomness + 5e-10:
        raise AssertionError("the finite-root optimizer violates the randomness budget")

    return PositiveFiniteRootResult(
        optimizer=optimizer,
        r_task=r_task,
        r_upper=r_upper,
        active_boundary=active_boundary,
        task_root_used=task_root_used,
        randomness_root_used=randomness_root_used,
        entropy_at_task=entropy_at_task,
        entropy_at_upper=entropy_at_upper,
    )


def symbolic_finite_root_checks() -> None:
    """Verify the concavity and task-monotonicity identities with SymPy."""

    q, y, r = sp.symbols("q y r", positive=True)
    beta = 1 - y - (1 - q) * r
    alpha = y - q * r
    natural_entropy = -r * sp.log(r) - beta * sp.log(beta) - alpha * sp.log(alpha)
    first = sp.diff(natural_entropy, r)
    expected_first = -sp.log(r) + (1 - q) * sp.log(beta) + q * sp.log(alpha)
    assert sp.simplify(first - expected_first) == 0
    second = sp.diff(natural_entropy, r, 2)
    expected_second = -(1 / r + (1 - q) ** 2 / beta + q**2 / alpha)
    assert sp.simplify(second - expected_second) == 0

    # Let s=P(S=1), z=P(S=1,Y=1).  The sign of G'(r) is the sign of
    # log[(y-z)(s-z)/(z(1-y-s+z))].  Its denominator exceeds its numerator by
    # z-sy=(1-2e)q(1-q)r, proving strict decrease for e<1/2 and r>0.
    s, z = sp.symbols("s z", real=True)
    numerator = (y - z) * (s - z)
    denominator = z * (1 - y - s + z)
    assert sp.expand(denominator - numerator - (z - s * y)) == 0


def random_full_schedule_grid_audit(
    seed: int = 2026080905, trials: int = 400, grid: int = 2001
) -> None:
    """Check that no sampled full K3/K4 positive schedule beats the exact root."""

    rng = random.Random(seed)
    solved = 0
    infeasible = 0
    checked_grid_points = 0
    for _ in range(trials):
        q = rng.uniform(0.02, 0.98)
        y = rng.uniform(0.02, 0.98)
        error = rng.uniform(0.0, 0.5)
        _, r_target = r_bounds(q, y)
        rho = rng.uniform(0.0, 1.15 * r_target)
        rate = rho * hb(q)
        r_upper = min(r_target, rho)
        g_zero = output_task_entropy(q, y, error, 0.0)
        g_upper = output_task_entropy(q, y, error, r_upper)
        task = rng.uniform(max(0.0, g_upper - 0.03), min(1.0, g_zero + 0.03))
        randomness = rng.uniform(0.0, 1.65)
        result = positive_finite_root_drc(
            q, y, error, rate, task, randomness, max_modes=4
        )
        if result is None:
            infeasible += 1
        else:
            solved += 1

        best_grid_r: float | None = None
        for index in range(grid):
            r = r_upper * index / max(grid - 1, 1)
            for max_modes in (3, 4):
                candidate = fixed_r_root_free_candidate(
                    q, y, error, r, rate, task, randomness, max_modes
                )
                checked_grid_points += 1
                if candidate is not None:
                    best_grid_r = r if best_grid_r is None else max(best_grid_r, r)
        if best_grid_r is not None:
            if result is None:
                raise AssertionError("the finite-root solver missed a grid-feasible point")
            if best_grid_r > result.optimizer.r + 5e-9:
                raise AssertionError("a full-schedule grid point beats the finite-root result")

    print(
        "Positive finite-root versus full-schedule grid audit: "
        f"{trials} budgets, {checked_grid_points} K3/K4 points, "
        f"{solved} feasible and {infeasible} infeasible, 0 violations"
    )


def main() -> None:
    symbolic_finite_root_checks()
    print("Positive finite-root symbolic identities: PASS")
    random_full_schedule_grid_audit()


if __name__ == "__main__":
    main()
