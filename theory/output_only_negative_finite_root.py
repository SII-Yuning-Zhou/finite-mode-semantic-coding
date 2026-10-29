"""Finite-root negative-branch and full forward DRC for Bernoulli model O.

Write u=-r>=0 and c=w_I on the negative-correlation branch.  The schedule is

    w(u,c) = (c, u+c, delta(u)-c, gamma(u)-c),

where delta=1-y-q*u and gamma=y-(1-q)*u.  At fixed u, strict concavity in c
reduces K=4 feasibility to the identity-absent endpoint c=0 and the
rate-truncated cap.  The cap is either a K=3 target face or the K=4 rate face
c=(rho-u)/2.

Every resulting branch has affine weights in u, so its entropy is strictly
concave.  On each closed branch interval the earliest feasible u is therefore
the left endpoint or one unique descending entropy root.  This yields an exact
finite-root solver for K in {3,4}.  Combining it with the positive theorem gives
the full two-branch forward DRC without an outer grid.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import sympy as sp

from .output_only_positive_finite_root import (
    PositiveFiniteRootResult,
    positive_finite_root_drc,
)
from .output_only_rb_theory_audit import (
    Point,
    TOL,
    entropy,
    hb,
    output_task_entropy,
    point,
    r_bounds,
)
from .output_only_root_free_drc import fixed_r_root_free_candidate


@dataclass(frozen=True)
class ScalarInterval:
    left: float
    right: float


@dataclass(frozen=True)
class NegativeEntropyBranch:
    name: str
    interval: ScalarInterval


@dataclass(frozen=True)
class NegativeBranchCandidate:
    name: str
    interval: ScalarInterval
    optimizer: Point
    root_used: bool
    entropy_left: float
    entropy_right: float


@dataclass(frozen=True)
class NegativeFiniteRootResult:
    optimizer: Point
    u_task: float
    u_upper: float
    candidates: tuple[NegativeBranchCandidate, ...]


@dataclass(frozen=True)
class FullFiniteRootResult:
    branch: str
    optimizer: Point
    positive_result: PositiveFiniteRootResult | None
    negative_result: NegativeFiniteRootResult | None


def negative_channel_probabilities(
    q: float, y: float, u: float
) -> tuple[float, float]:
    """Return delta and gamma for the negative-correlation channel."""

    delta = 1.0 - y - q * u
    gamma = y - (1.0 - q) * u
    return delta, gamma


def negative_weights(
    q: float, y: float, rho: float, u: float, branch: str
) -> tuple[float, float, float, float]:
    """Return mode weights on one negative finite-root branch."""

    delta, gamma = negative_channel_probabilities(q, y, u)
    if branch == "identity-absent":
        c = 0.0
    elif branch == "rate":
        c = (rho - u) / 2.0
    elif branch == "target-w0-zero":
        c = delta
    elif branch == "target-w1-zero":
        c = gamma
    else:
        raise ValueError(f"unknown negative entropy branch: {branch}")
    return c, u + c, delta - c, gamma - c


def negative_branch_entropy(
    q: float, y: float, rho: float, u: float, branch: str
) -> float:
    weights = negative_weights(q, y, rho, u, branch)
    if min(weights) < -3e-9:
        raise ValueError(f"u is outside the {branch} branch interval")
    return entropy(weights)


def _clip_affine_leq(
    interval: ScalarInterval | None, slope: float, intercept: float
) -> ScalarInterval | None:
    """Intersect an interval with slope*u+intercept <= 0."""

    if interval is None:
        return None
    left, right = interval.left, interval.right
    if abs(slope) <= 1e-14:
        return interval if intercept <= 2e-12 else None
    boundary = -intercept / slope
    if slope > 0.0:
        right = min(right, boundary)
    else:
        left = max(left, boundary)
    if left > right + 2e-11:
        return None
    left = min(max(left, interval.left), interval.right)
    right = min(max(right, interval.left), interval.right)
    return ScalarInterval(left, right)


def negative_branch_intervals(
    q: float, y: float, rho: float, u_task: float, u_upper: float
) -> tuple[NegativeEntropyBranch, ...]:
    """Return the at-most-four affine branch intervals.

    The target branches are available when their target cap is reachable by
    the payload radius.  The rate branch is available when the rate cap occurs
    before both target caps.
    """

    base = ScalarInterval(u_task, u_upper)
    branches = [NegativeEntropyBranch("identity-absent", base)]
    d = 1.0 - 2.0 * q

    # Rate: rho <= a_delta(u) and rho <= a_gamma(u).
    rate_interval = _clip_affine_leq(base, -d, rho - 2.0 * (1.0 - y))
    rate_interval = _clip_affine_leq(rate_interval, d, rho - 2.0 * y)
    if rate_interval is not None:
        branches.append(NegativeEntropyBranch("rate", rate_interval))

    # Target w0=0: delta <= gamma and a_delta(u) <= rho.
    target_w0 = _clip_affine_leq(base, d, 1.0 - 2.0 * y)
    target_w0 = _clip_affine_leq(
        target_w0, d, 2.0 * (1.0 - y) - rho
    )
    if target_w0 is not None:
        branches.append(NegativeEntropyBranch("target-w0-zero", target_w0))

    # Target w1=0: gamma <= delta and a_gamma(u) <= rho.
    target_w1 = _clip_affine_leq(base, -d, 2.0 * y - 1.0)
    target_w1 = _clip_affine_leq(target_w1, -d, 2.0 * y - rho)
    if target_w1 is not None:
        branches.append(NegativeEntropyBranch("target-w1-zero", target_w1))

    return tuple(branches)


def _bisect_negative_task_floor(
    q: float, y: float, error: float, task: float, right: float
) -> float:
    """Solve G(-u)=C where G(-u) decreases strictly with u."""

    left = 0.0
    f_left = output_task_entropy(q, y, error, -left) - task
    f_right = output_task_entropy(q, y, error, -right) - task
    if abs(f_left) <= 2e-13:
        return left
    if abs(f_right) <= 2e-13:
        return right
    if not (f_left > 0.0 and f_right < 0.0):
        raise ValueError("the negative task root is not bracketed")
    for _ in range(180):
        middle = (left + right) / 2.0
        f_middle = output_task_entropy(q, y, error, -middle) - task
        if f_middle > 0.0:
            left = middle
        else:
            right = middle
    return (left + right) / 2.0


def _bisect_descending_entropy_root(
    q: float,
    y: float,
    rho: float,
    randomness: float,
    branch: NegativeEntropyBranch,
) -> float:
    """Find the unique level root from superlevel-left to sublevel-right."""

    left, right = branch.interval.left, branch.interval.right
    f_left = negative_branch_entropy(q, y, rho, left, branch.name) - randomness
    f_right = negative_branch_entropy(q, y, rho, right, branch.name) - randomness
    if abs(f_left) <= 2e-13:
        return left
    if abs(f_right) <= 2e-13:
        return right
    if not (f_left > 0.0 and f_right < 0.0):
        raise ValueError("the descending entropy root is not bracketed")
    for _ in range(180):
        middle = (left + right) / 2.0
        f_middle = (
            negative_branch_entropy(q, y, rho, middle, branch.name)
            - randomness
        )
        if f_middle > 0.0:
            left = middle
        else:
            right = middle
    return (left + right) / 2.0


def _earliest_branch_candidate(
    q: float,
    y: float,
    error: float,
    rho: float,
    rate: float,
    task: float,
    randomness: float,
    branch: NegativeEntropyBranch,
) -> NegativeBranchCandidate | None:
    left, right = branch.interval.left, branch.interval.right
    entropy_left = negative_branch_entropy(q, y, rho, left, branch.name)
    entropy_right = negative_branch_entropy(q, y, rho, right, branch.name)

    if entropy_left <= randomness + 2e-11:
        u_star = left
        root_used = False
    elif entropy_right <= randomness + 2e-11:
        u_star = _bisect_descending_entropy_root(
            q, y, rho, randomness, branch
        )
        root_used = True
    else:
        # Strict concavity puts every interior entropy above the endpoint chord.
        return None

    weights = negative_weights(q, y, rho, u_star, branch.name)
    c = weights[0]
    optimizer = point(q, y, error, -u_star, u_star + c)
    if min(optimizer.weights) < -5e-9:
        raise AssertionError("negative branch candidate has a negative weight")
    if optimizer.payload > rate + 5e-9:
        raise AssertionError("negative branch candidate violates payload")
    if optimizer.task > task + 5e-9:
        raise AssertionError("negative branch candidate violates task")
    if optimizer.randomness > randomness + 5e-9:
        raise AssertionError("negative branch candidate violates randomness")
    return NegativeBranchCandidate(
        name=branch.name,
        interval=branch.interval,
        optimizer=optimizer,
        root_used=root_used,
        entropy_left=entropy_left,
        entropy_right=entropy_right,
    )


def negative_finite_root_drc(
    q: float,
    y: float,
    error: float,
    rate: float,
    task: float,
    randomness: float,
    max_modes: int = 4,
) -> NegativeFiniteRootResult | None:
    """Solve the exact r<=0 branch for K=3 or K=4."""

    if max_modes not in (3, 4):
        raise ValueError("the negative finite-root theorem applies to K=3 or K=4")
    if not 0.0 < q < 1.0 or not 0.0 < y < 1.0:
        raise ValueError("negative_finite_root_drc requires 0<q,y<1")
    if not 0.0 <= error <= 0.5:
        raise ValueError("the task ordering requires 0<=error<=1/2")
    if rate < -TOL or randomness < -TOL:
        return None

    r_lower, _ = r_bounds(q, y)
    u_target = -r_lower
    rho = max(0.0, rate) / hb(q)
    u_upper = min(u_target, rho)
    if u_upper < -TOL:
        return None
    u_upper = max(0.0, u_upper)

    task_at_zero = output_task_entropy(q, y, error, 0.0)
    task_at_upper = output_task_entropy(q, y, error, -u_upper)
    if task_at_upper > task + 2e-11:
        return None
    if task_at_zero <= task + 2e-11:
        u_task = 0.0
    else:
        u_task = _bisect_negative_task_floor(
            q, y, error, task, u_upper
        )

    branches = negative_branch_intervals(q, y, rho, u_task, u_upper)
    if max_modes == 3:
        branches = tuple(branch for branch in branches if branch.name != "rate")

    candidates: list[NegativeBranchCandidate] = []
    for branch in branches:
        candidate = _earliest_branch_candidate(
            q, y, error, rho, rate, task, randomness, branch
        )
        if candidate is None:
            continue
        if candidate.optimizer.support > max_modes:
            # The K4 rate branch can become K3 at a shared endpoint; all genuine
            # K3 endpoint schedules are already represented by target/map faces.
            continue
        candidates.append(candidate)

    if not candidates:
        return None
    candidates.sort(
        key=lambda candidate: (
            -candidate.optimizer.r,
            candidate.optimizer.support,
            candidate.optimizer.payload,
        )
    )
    optimizer = candidates[0].optimizer
    return NegativeFiniteRootResult(
        optimizer=optimizer,
        u_task=u_task,
        u_upper=u_upper,
        candidates=tuple(candidates),
    )


def full_finite_root_drc(
    q: float,
    y: float,
    error: float,
    rate: float,
    task: float,
    randomness: float,
    max_modes: int = 4,
) -> FullFiniteRootResult | None:
    """Solve the complete two-branch forward DRC for K=3 or K=4."""

    positive = positive_finite_root_drc(
        q, y, error, rate, task, randomness, max_modes=max_modes
    )
    if positive is not None:
        return FullFiniteRootResult(
            branch="positive",
            optimizer=positive.optimizer,
            positive_result=positive,
            negative_result=None,
        )

    negative = negative_finite_root_drc(
        q, y, error, rate, task, randomness, max_modes=max_modes
    )
    if negative is None:
        return None
    return FullFiniteRootResult(
        branch="negative",
        optimizer=negative.optimizer,
        positive_result=None,
        negative_result=negative,
    )


def symbolic_negative_finite_root_checks() -> None:
    """Verify all branch second derivatives and affine payload identities."""

    q, y, u, rho = sp.symbols("q y u rho", real=True)
    delta = 1 - y - q * u
    gamma = y - (1 - q) * u
    c_rate = (rho - u) / 2
    schedules = {
        "identity-absent": (0, u, delta, gamma),
        "rate": (
            c_rate,
            u + c_rate,
            delta - c_rate,
            gamma - c_rate,
        ),
        "target-w0-zero": (delta, u + delta, 0, gamma - delta),
        "target-w1-zero": (gamma, u + gamma, delta - gamma, 0),
    }

    for name, weights in schedules.items():
        assert sp.simplify(sum(weights) - 1) == 0
        positive_weights = tuple(weight for weight in weights if weight != 0)
        natural_entropy = -sum(
            weight * sp.log(weight) for weight in positive_weights
        )
        second = sp.diff(natural_entropy, u, 2)
        expected = -sum(
            sp.diff(weight, u) ** 2 / weight for weight in positive_weights
        )
        assert sp.simplify(second - expected) == 0, name

    rate_payload = -u + 2 * (u + c_rate)
    assert sp.simplify(rate_payload - rho) == 0
    a_delta = u + 2 * delta
    a_gamma = u + 2 * gamma
    assert sp.expand(a_delta - (2 * (1 - y) + (1 - 2 * q) * u)) == 0
    assert sp.expand(a_gamma - (2 * y - (1 - 2 * q) * u)) == 0

    # With z=sy-k*u, the sign identity reverses relative to r>0:
    # denominator-numerator=z-sy=-k*u<0, hence G(-u) decreases with u.
    s, z = sp.symbols("s z", real=True)
    numerator = (y - z) * (s - z)
    denominator = z * (1 - y - s + z)
    assert sp.expand(denominator - numerator - (z - s * y)) == 0


def negative_k4_separation_regression() -> None:
    """Reproduce the P0015 strict negative K4 gain with the exact solver."""

    q = 0.326236854
    y = 0.387768557
    error = 0.328538306
    rate = 0.564340860
    task = 0.975871284
    randomness = 1.335539723
    k3 = full_finite_root_drc(
        q, y, error, rate, task, randomness, max_modes=3
    )
    k4 = full_finite_root_drc(
        q, y, error, rate, task, randomness, max_modes=4
    )
    if k3 is None or k4 is None:
        raise AssertionError("the negative separation witness became infeasible")
    if not k4.optimizer.r > k3.optimizer.r + 1e-4:
        raise AssertionError("the exact solver lost the strict negative K4 gain")
    print("Exact negative-branch K4 separation")
    print(f"  K3={k3.optimizer}")
    print(f"  K4={k4.optimizer}")
    print(f"  distortion gap={k3.optimizer.distortion-k4.optimizer.distortion:.12f}")


def random_full_grid_audit(
    seed: int = 2026080906, trials: int = 400, grid: int = 2001
) -> None:
    """Check the exact two-branch result against full K3/K4 r grids."""

    rng = random.Random(seed)
    checked = 0
    feasible = 0
    for _ in range(trials):
        q = rng.uniform(0.02, 0.98)
        y = rng.uniform(0.02, 0.98)
        error = rng.uniform(0.0, 0.5)
        rate = rng.uniform(0.0, 1.05) * hb(q)
        task = rng.uniform(0.55, 1.0)
        randomness = rng.uniform(0.0, 1.75)
        lo, hi = r_bounds(q, y)

        for max_modes in (3, 4):
            exact = full_finite_root_drc(
                q, y, error, rate, task, randomness, max_modes
            )
            if exact is not None:
                feasible += 1
            best_grid_r: float | None = None
            for index in range(grid):
                r = lo + (hi - lo) * index / max(grid - 1, 1)
                candidate = fixed_r_root_free_candidate(
                    q,
                    y,
                    error,
                    r,
                    rate,
                    task,
                    randomness,
                    max_modes,
                )
                checked += 1
                if candidate is not None:
                    best_grid_r = (
                        r if best_grid_r is None else max(best_grid_r, r)
                    )
            if best_grid_r is not None:
                if exact is None:
                    raise AssertionError("finite-root solver missed a grid point")
                if best_grid_r > exact.optimizer.r + 5e-9:
                    raise AssertionError("a full r-grid point beats the exact solver")

    print(
        "Full finite-root versus K3/K4 r-grid audit: "
        f"{2*trials} problems, {checked} fixed-r points, "
        f"{feasible} feasible exact results, 0 violations"
    )


def main() -> None:
    symbolic_negative_finite_root_checks()
    print("Negative finite-root symbolic identities: PASS")
    negative_k4_separation_regression()
    random_full_grid_audit()


if __name__ == "__main__":
    main()
