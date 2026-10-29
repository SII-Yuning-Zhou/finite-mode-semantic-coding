"""Numerical audit for mode-limited target-distribution semantic coding.

The four deterministic binary codec modes are
    0: identity, 1: flip, 2: constant zero, 3: constant one.

This script does not replace a proof.  It checks the algebraic reduction,
implements the exact finite candidate/face optimizers for K <= 3, and
implements the finite-candidate K = 4 optimizer proved in the accompanying
theory memo.  The older one-dimensional grid is retained only as a regression
oracle.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np


def hb(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def entropy(weights: np.ndarray) -> float:
    positive = weights[weights > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


@dataclass(frozen=True)
class Metrics:
    weights: tuple[float, float, float, float]
    distortion: float
    payload_rate: float
    common_randomness: float
    task_entropy_mode_aware: float
    task_entropy_output_only: float
    support: int


@dataclass(frozen=True)
class Candidate:
    name: str
    metric: Metrics


@dataclass(frozen=True)
class ProductBalancePoint:
    """All-positive point satisfying w_I w_F = w_0 w_1 at fixed a."""

    r: float
    b: float
    entropy: float
    a_slope: float
    entropy_r_slope: float
    weights: tuple[float, float, float, float]


def weights_from_ab(qx: float, qy: float, a: float, b: float) -> np.ndarray:
    """Return (identity, flip, zero, one) weights.

    a is the total probability of an information-preserving mode and b is the
    flip-mode probability.  Exact P(Y=1)=qy and sum(weights)=1 are imposed.
    """

    w1 = a - b
    w2 = b
    w4 = qy - qx * a - (1.0 - 2.0 * qx) * b
    w3 = 1.0 - a - w4
    return np.array([w1, w2, w3, w4], dtype=float)


def output_only_task_entropy(
    qx: float, crossover: float, weights: np.ndarray
) -> float:
    maps = ((0, 1), (1, 0), (0, 0), (1, 1))
    joint = np.zeros((2, 2), dtype=float)  # rows S, columns Y
    for weight, mapping in zip(weights, maps):
        for x, px in ((0, 1.0 - qx), (1, qx)):
            for noise, pn in ((0, 1.0 - crossover), (1, crossover)):
                s = x ^ noise
                y = mapping[x]
                joint[s, y] += weight * px * pn

    result = 0.0
    for y in (0, 1):
        py = float(joint[:, y].sum())
        if py > 0.0:
            result += py * hb(float(joint[1, y] / py))
    return result


def evaluate(qx: float, qy: float, crossover: float, a: float, b: float) -> Metrics:
    weights = weights_from_ab(qx, qy, a, b)
    q_s = qx + crossover - 2.0 * qx * crossover
    h_mode = a * hb(crossover) + (1.0 - a) * hb(q_s)
    distortion = (
        weights[1] + qx * weights[2] + (1.0 - qx) * weights[3]
    )
    return Metrics(
        weights=tuple(float(x) for x in weights),
        distortion=float(distortion),
        payload_rate=float(a * hb(qx)),
        common_randomness=entropy(weights),
        task_entropy_mode_aware=float(h_mode),
        task_entropy_output_only=output_only_task_entropy(qx, crossover, weights),
        support=int(np.count_nonzero(weights > 1e-10)),
    )


def optimize_balanced_source(
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
) -> Candidate | None:
    """Exact qx=1/2 optimum after merging flip into identity.

    Identity and flip have the same output marginal, payload, and mode-aware
    task entropy when the source is balanced.  Moving all flip probability
    to identity reduces both Hamming distortion and schedule entropy.  Hence
    K=3 is always sufficient and the remaining scalar problem maximizes the
    identity mass a under the resource budgets.
    """

    if not 0.0 <= qy <= 1.0:
        raise ValueError("target Bernoulli parameter qy must lie in [0,1]")
    if rate < -1e-14:
        return None
    h_noise = hb(crossover)
    if 1.0 > h_noise + 1e-14:
        lower = max(
            0.0, (1.0 - task_entropy) / (1.0 - h_noise)
        )
    else:
        lower = 0.0 if task_entropy >= 1.0 - 1e-14 else math.inf
    upper = min(2.0 * qy, 2.0 * (1.0 - qy), rate)
    if lower > upper + 1e-12:
        return None
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    def schedule_entropy(a: float) -> float:
        return entropy(
            np.array(
                [a, 1.0 - qy - a / 2.0, qy - a / 2.0],
                dtype=float,
            )
        )

    if schedule_entropy(upper) <= randomness_rate + 1e-12:
        a_star = upper
    elif schedule_entropy(lower) > randomness_rate + 1e-12:
        return None
    else:
        left, right = lower, upper
        for _ in range(180):
            middle = (left + right) / 2.0
            if schedule_entropy(middle) <= randomness_rate:
                left = middle
            else:
                right = middle
        a_star = (left + right) / 2.0

    return Candidate(
        "qhalf-K3-scalar",
        evaluate(0.5, qy, crossover, a_star, 0.0),
    )


def optimize_deterministic_source(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
) -> Candidate | None:
    """Exact qx in {0,1} optimum after merging support-equivalent modes."""

    if qx not in (0.0, 1.0):
        raise ValueError("deterministic-source optimizer requires qx=0 or 1")
    if not 0.0 <= qy <= 1.0:
        raise ValueError("target Bernoulli parameter qy must lie in [0,1]")
    if rate < -1e-14:
        return None
    if task_entropy + 1e-12 < hb(crossover):
        return None
    if randomness_rate + 1e-12 < hb(qy):
        return None
    return Candidate(
        "deterministic-source-K2",
        evaluate(qx, qy, crossover, 0.0, 0.0),
    )


def closed_form_unlimited_mode_aware(
    qx: float, qy: float, crossover: float, rate: float, task_entropy: float
) -> Metrics | None:
    """Exact candidate without a common-randomness entropy constraint.

    Valid for 0 < qx < 1/2 and 0 <= qy <= 1/2.  The expression follows by
    eliminating the target-marginal and normalization constraints.
    """

    h_x = hb(qx)
    q_s = qx + crossover - 2.0 * qx * crossover
    h_good = hb(crossover)
    h_bad = hb(q_s)

    if h_bad > h_good + 1e-14:
        a_min = max(0.0, (h_bad - task_entropy) / (h_bad - h_good))
    else:
        a_min = 0.0 if task_entropy >= h_bad else math.inf

    a_max_target = min(1.0, qy / qx)
    a_max = min(a_max_target, rate / h_x if h_x > 0.0 else 0.0)
    if a_min > a_max + 1e-12:
        return None

    a_ot = min(qy / qx, (1.0 - qy) / (1.0 - qx))
    a_star = min(max(a_ot, a_min), a_max)
    threshold = (1.0 - qy) / (1.0 - qx)
    b_star = max(
        0.0,
        ((1.0 - qx) * a_star - (1.0 - qy)) / (1.0 - 2.0 * qx),
    )
    return evaluate(qx, qy, crossover, a_star, b_star)


def pair_candidates(qx: float, qy: float, crossover: float) -> list[Candidate]:
    """All exact-target candidates with at most two active codec modes."""

    maps = (
        ("identity", qx),
        ("flip", 1.0 - qx),
        ("zero", 0.0),
        ("one", 1.0),
    )
    result: list[Candidate] = []
    tolerance = 1e-12

    # Single-mode candidates.
    for index, (name, output_one_probability) in enumerate(maps):
        if abs(output_one_probability - qy) <= tolerance:
            weights = np.zeros(4)
            weights[index] = 1.0
            a = float(weights[0] + weights[1])
            b = float(weights[1])
            result.append(Candidate(name, evaluate(qx, qy, crossover, a, b)))

    # Two-mode candidates.  The target marginal fixes the mixing weight.
    for i, j in combinations(range(4), 2):
        name_i, p_i = maps[i]
        name_j, p_j = maps[j]
        if abs(p_i - p_j) <= tolerance:
            continue
        lam = (qy - p_j) / (p_i - p_j)
        if lam < -tolerance or lam > 1.0 + tolerance:
            continue
        lam = min(1.0, max(0.0, lam))
        weights = np.zeros(4)
        weights[i] = lam
        weights[j] = 1.0 - lam
        a = float(weights[0] + weights[1])
        b = float(weights[1])
        result.append(
            Candidate(f"{name_i}+{name_j}", evaluate(qx, qy, crossover, a, b))
        )
    return result


def optimize_k2(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
) -> Candidate | None:
    feasible: list[Candidate] = []
    for candidate in pair_candidates(qx, qy, crossover):
        metric = candidate.metric
        if metric.payload_rate > rate + 1e-12:
            continue
        if metric.task_entropy_mode_aware > task_entropy + 1e-12:
            continue
        if metric.common_randomness > randomness_rate + 1e-12:
            continue
        feasible.append(candidate)
    return min(feasible, key=lambda item: item.metric.distortion, default=None)


def face_weights(face: str, qx: float, qy: float, a: float) -> np.ndarray:
    """Weights on one of the four three-mode faces."""

    c = 1.0 - 2.0 * qx
    if face == "134":  # flip is absent
        return np.array(
            [a, 0.0, 1.0 - qy - (1.0 - qx) * a, qy - qx * a]
        )
    if face == "124":  # zero is absent
        return np.array(
            [
                (1.0 - qy - qx * a) / c,
                ((1.0 - qx) * a - (1.0 - qy)) / c,
                0.0,
                1.0 - a,
            ]
        )
    if face == "123":  # one is absent
        return np.array(
            [
                ((1.0 - qx) * a - qy) / c,
                (qy - qx * a) / c,
                1.0 - a,
                0.0,
            ]
        )
    if face == "234":  # identity is absent
        return np.array(
            [0.0, a, 1.0 - qy - qx * a, qy - (1.0 - qx) * a]
        )
    raise ValueError(f"unknown face: {face}")


def face_base_interval(face: str, qx: float, qy: float) -> tuple[float, float] | None:
    """Exact target/nonnegativity interval before rate and task constraints.

    The formulas are valid for 0 < qx < 1/2 and 0 <= qy <= 1.
    """

    if face == "134":
        interval = (
            0.0,
            min(1.0, qy / qx, (1.0 - qy) / (1.0 - qx)),
        )
    elif face == "124":
        interval = (
            (1.0 - qy) / (1.0 - qx),
            min(1.0, (1.0 - qy) / qx),
        )
    elif face == "123":
        interval = (qy / (1.0 - qx), min(1.0, qy / qx))
    elif face == "234":
        interval = (
            0.0,
            min(1.0, qy / (1.0 - qx), (1.0 - qy) / qx),
        )
    else:
        raise ValueError(f"unknown face: {face}")
    if interval[0] > interval[1] + 1e-12:
        return None
    return interval


def face_objective_decreases(face: str) -> bool:
    return face in {"134", "123"}


def metric_from_face(
    face: str, qx: float, qy: float, crossover: float, a: float
) -> Metrics:
    weights = face_weights(face, qx, qy, a)
    b = float(weights[1])
    return evaluate(qx, qy, crossover, a, b)


def _entropy_root(
    face: str,
    qx: float,
    qy: float,
    crossover: float,
    left: float,
    right: float,
    target: float,
) -> float:
    def f(value: float) -> float:
        return (
            metric_from_face(face, qx, qy, crossover, value).common_randomness
            - target
        )

    f_left = f(left)
    f_right = f(right)
    if abs(f_left) < 1e-13:
        return left
    if abs(f_right) < 1e-13:
        return right
    if f_left * f_right > 0.0:
        raise ValueError("entropy root is not bracketed")
    for _ in range(180):
        middle = (left + right) / 2.0
        f_middle = f(middle)
        if f_left * f_middle <= 0.0:
            right = middle
            f_right = f_middle
        else:
            left = middle
            f_left = f_middle
    return (left + right) / 2.0


def optimize_face(
    face: str,
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
) -> Candidate | None:
    if not 0.0 < qx < 0.5:
        raise ValueError("K=3 face optimizer requires 0 < qx < 1/2")
    if not 0.0 <= qy <= 1.0:
        raise ValueError("target Bernoulli parameter qy must lie in [0,1]")

    base = face_base_interval(face, qx, qy)
    if base is None:
        return None

    q_s = qx + crossover - 2.0 * qx * crossover
    h_good = hb(crossover)
    h_bad = hb(q_s)
    if h_bad > h_good + 1e-14:
        a_min_task = max(0.0, (h_bad - task_entropy) / (h_bad - h_good))
    else:
        a_min_task = 0.0 if task_entropy >= h_bad - 1e-14 else math.inf
    a_max_rate = rate / hb(qx)
    left = max(base[0], a_min_task)
    right = min(base[1], a_max_rate)
    if left > right + 1e-12:
        return None
    left = min(max(left, base[0]), base[1])
    right = min(max(right, base[0]), base[1])

    h_left = metric_from_face(face, qx, qy, crossover, left).common_randomness
    h_right = metric_from_face(face, qx, qy, crossover, right).common_randomness
    decreases = face_objective_decreases(face)

    if decreases and h_right <= randomness_rate + 1e-12:
        chosen = right
    elif (not decreases) and h_left <= randomness_rate + 1e-12:
        chosen = left
    else:
        # Concavity implies infeasibility if both endpoint entropies exceed Rc.
        if h_left > randomness_rate + 1e-12 and h_right > randomness_rate + 1e-12:
            return None
        if decreases:
            # The right endpoint is blocked while the left endpoint is
            # feasible.  Strict concavity gives exactly one crossing.
            chosen = _entropy_root(
                face, qx, qy, crossover, left, right, randomness_rate
            )
        else:
            # The left endpoint is blocked while the right endpoint is
            # feasible.  Strict concavity gives exactly one crossing.
            chosen = _entropy_root(
                face, qx, qy, crossover, left, right, randomness_rate
            )
    return Candidate(
        f"F{face}", metric_from_face(face, qx, qy, crossover, chosen)
    )


def optimize_k3(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
) -> Candidate | None:
    candidates = [
        candidate
        for face in ("134", "124", "123", "234")
        if (
            candidate := optimize_face(
                face,
                qx,
                qy,
                crossover,
                rate,
                task_entropy,
                randomness_rate,
            )
        )
        is not None
    ]
    return min(candidates, key=lambda item: item.metric.distortion, default=None)


def feasible_a_interval(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
) -> tuple[float, float] | None:
    """Feasible interval of the information-preserving mass a."""

    if not 0.0 < qx < 0.5:
        raise ValueError("K=4 optimizer requires 0 < qx < 1/2")
    if not 0.0 <= qy <= 1.0:
        raise ValueError("target Bernoulli parameter qy must lie in [0,1]")

    q_s = qx + crossover - 2.0 * qx * crossover
    h_good = hb(crossover)
    h_bad = hb(q_s)
    if h_bad > h_good + 1e-14:
        lower = max(0.0, (h_bad - task_entropy) / (h_bad - h_good))
    else:
        lower = 0.0 if task_entropy >= h_bad - 1e-14 else math.inf
    # Exact-target feasibility requires aqx <= qy <= 1-aqx.  The second
    # inequality is active for targets above 1/2 and was missing in the first
    # numerical draft.
    upper = min(1.0, qy / qx, (1.0 - qy) / qx, rate / hb(qx))
    if lower > upper + 1e-12:
        return None
    return lower, upper


def feasible_b_interval(qx: float, qy: float, a: float) -> tuple[float, float] | None:
    """Exact-target/nonnegativity interval of b for a fixed a."""

    c = 1.0 - 2.0 * qx
    lower = max(0.0, ((1.0 - qx) * a - (1.0 - qy)) / c)
    upper = min(a, (qy - qx * a) / c)
    if lower > upper + 1e-12:
        return None
    return lower, upper


def optimize_b_at_fixed_a(
    qx: float,
    qy: float,
    crossover: float,
    a: float,
    randomness_rate: float,
) -> Metrics | None:
    """Exact fixed-a optimum under H(V) <= Rc.

    Distortion is increasing in b.  Entropy is concave in b.  Consequently,
    the solution is b_L if that endpoint obeys the randomness budget; it is
    infeasible if both endpoints violate it; otherwise it is the unique root
    H(w(a,b)) = Rc between the endpoints.
    """

    interval = feasible_b_interval(qx, qy, a)
    if interval is None:
        return None
    left, right = interval

    def h(value: float) -> float:
        return entropy(weights_from_ab(qx, qy, a, value))

    h_left = h(left)
    if h_left <= randomness_rate + 1e-12:
        return evaluate(qx, qy, crossover, a, left)
    h_right = h(right)
    if h_right > randomness_rate + 1e-12:
        return None

    # Here f(left) > 0 >= f(right).  Concavity gives the unique descending
    # crossing and the smallest feasible b.
    for _ in range(180):
        middle = (left + right) / 2.0
        if h(middle) > randomness_rate:
            left = middle
        else:
            right = middle
    return evaluate(qx, qy, crossover, a, (left + right) / 2.0)


def optimize_k4_numeric(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
    grid: int = 4001,
    refinements: int = 4,
) -> Candidate | None:
    """Grid-reference minimization of the one-dimensional K=4 reduction.

    A global grid is followed by local grid refinements.  The inner b problem
    is solved exactly up to bisection tolerance by ``optimize_b_at_fixed_a``.
    """

    interval = feasible_a_interval(qx, qy, crossover, rate, task_entropy)
    if interval is None:
        return None
    global_left, global_right = interval
    search_left, search_right = global_left, global_right
    best: Metrics | None = None
    best_a: float | None = None

    for refinement in range(refinements + 1):
        count = grid if refinement == 0 else 501
        for a in np.linspace(search_left, search_right, count):
            metric = optimize_b_at_fixed_a(
                qx, qy, crossover, float(a), randomness_rate
            )
            if metric is None:
                continue
            if best is None or metric.distortion < best.distortion:
                best = metric
                best_a = float(a)
        if best_a is None or search_right <= search_left:
            break
        step = (search_right - search_left) / max(count - 1, 1)
        search_left = max(global_left, best_a - 2.0 * step)
        search_right = min(global_right, best_a + 2.0 * step)

    if best is None:
        return None
    return Candidate("K4-one-dimensional", best)


def _bisect_level(
    function,
    left: float,
    right: float,
    level: float,
    iterations: int = 180,
) -> float | None:
    """Return a bracketed root of function(x)=level, including endpoints."""

    f_left = function(left) - level
    f_right = function(right) - level
    tolerance = 2e-13
    if abs(f_left) <= tolerance:
        return left
    if abs(f_right) <= tolerance:
        return right
    if f_left * f_right > 0.0:
        return None
    for _ in range(iterations):
        middle = (left + right) / 2.0
        f_middle = function(middle) - level
        if f_left * f_middle <= 0.0:
            right = middle
            f_right = f_middle
        else:
            left = middle
            f_left = f_middle
    return (left + right) / 2.0


def stationary_k4_candidates(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
) -> list[Candidate]:
    """All admissible all-positive interior stationary K=4 candidates.

    At such a candidate H(V)=Rc and w_I w_F = w_0 w_1.  With r=a-2b,
    define A=1-qy-(1-qx)r and B=qy-qx r.  The weights factor as

        ((1-A)(1-B), A B, A(1-B), (1-A)B),

    and stationarity reduces to the strictly concave scalar equation
    hb(A(r))+hb(B(r))=Rc.  It therefore has at most two roots.
    """

    a_interval = feasible_a_interval(qx, qy, crossover, rate, task_entropy)
    if a_interval is None:
        return []
    a_left, a_right = a_interval

    r_left = max(-qy / (1.0 - qx), -(1.0 - qy) / qx)
    r_right = min((1.0 - qy) / (1.0 - qx), qy / qx)
    if r_left > r_right + 1e-14:
        return []

    def ab_from_r(r: float) -> tuple[float, float, float, float]:
        product_a = 1.0 - qy - (1.0 - qx) * r
        product_b = qy - qx * r
        b = product_a * product_b
        a = r + 2.0 * b
        return product_a, product_b, a, b

    def stationary_entropy(r: float) -> float:
        product_a, product_b, _, _ = ab_from_r(r)
        return hb(product_a) + hb(product_b)

    # Strict concavity gives a unique maximizer (apart from a degenerate
    # zero-length interval).  Ternary search locates it only for bracketing;
    # no outer optimization grid is used.
    peak_left, peak_right = r_left, r_right
    for _ in range(180):
        first = (2.0 * peak_left + peak_right) / 3.0
        second = (peak_left + 2.0 * peak_right) / 3.0
        if stationary_entropy(first) < stationary_entropy(second):
            peak_left = first
        else:
            peak_right = second
    r_peak = (peak_left + peak_right) / 2.0

    roots: list[float] = []
    for branch_left, branch_right in (
        (r_left, r_peak),
        (r_peak, r_right),
    ):
        root = _bisect_level(
            stationary_entropy,
            branch_left,
            branch_right,
            randomness_rate,
        )
        if root is not None and not any(abs(root - old) <= 2e-10 for old in roots):
            roots.append(root)

    result: list[Candidate] = []
    tolerance = 2e-9
    for index, root in enumerate(roots, start=1):
        _, _, a, b = ab_from_r(root)
        weights = weights_from_ab(qx, qy, a, b)

        # Weight-boundary points belong to K<=3 and are already covered by
        # optimize_k3.  Here we retain only genuine interior K=4 candidates.
        if np.min(weights) <= 1e-10:
            continue
        if not a_left + 1e-10 < a < a_right - 1e-10:
            continue

        metric = evaluate(qx, qy, crossover, a, b)
        if abs(metric.common_randomness - randomness_rate) > tolerance:
            continue

        # The entropy level can have another, larger-distortion branch at the
        # same a.  Keep a stationary root only if it is the smallest feasible
        # b selected by the exact fixed-a optimizer.
        inner = optimize_b_at_fixed_a(qx, qy, crossover, a, randomness_rate)
        if inner is None or abs(inner.weights[1] - b) > tolerance:
            continue
        result.append(Candidate(f"K4-stationary-{index}", metric))
    return result


def optimize_k4_finite_candidates(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
) -> Candidate | None:
    """Exact finite-candidate K=4 optimizer up to scalar root tolerance.

    The complete minimizing candidate set contains the globally optimal K<=3
    point and the exact fixed-a optima at both resource endpoints.  An
    all-positive interior stationary point must satisfy w_I w_F=w_0 w_1, but
    its second derivative on the selected descending entropy branch is
    strictly negative, so it is a local maximum and can be discarded.  No
    a-grid is used.
    """

    interval = feasible_a_interval(qx, qy, crossover, rate, task_entropy)
    if interval is None:
        return None
    left, right = interval

    candidates: list[Candidate] = []
    k3 = optimize_k3(
        qx, qy, crossover, rate, task_entropy, randomness_rate
    )
    if k3 is not None:
        candidates.append(Candidate("K4-boundary-" + k3.name, k3.metric))

    for name, a in (("left", left), ("right", right)):
        metric = optimize_b_at_fixed_a(
            qx, qy, crossover, a, randomness_rate
        )
        if metric is not None:
            candidates.append(Candidate(f"K4-a-{name}", metric))

    return min(candidates, key=lambda item: item.metric.distortion, default=None)


def product_balance_at_fixed_a(
    qx: float, qy: float, a: float
) -> ProductBalancePoint | None:
    """Return the feasible algebraic product-balance point at fixed a.

    With r=a-2b, product balance factors the four mode probabilities as

        ((1-A)(1-B), AB, A(1-B), (1-A)B),

    where A=1-qy-(1-qx)r and B=qy-qx*r.  The relevant branch is the plus
    square-root solution of

        a=2q(1-q)r^2+(1-2q)(1-2qy)r+2qy(1-qy).
    """

    if not 0.0 < qx < 0.5 or not 0.0 < qy < 1.0:
        return None
    c = 1.0 - 2.0 * qx
    product_q = qx * (1.0 - qx)
    discriminant = (
        c * c * (1.0 - 2.0 * qy) ** 2
        - 8.0 * product_q * (2.0 * qy * (1.0 - qy) - a)
    )
    if discriminant < -1e-13:
        return None
    discriminant = max(0.0, discriminant)
    r = (
        -c * (1.0 - 2.0 * qy) + math.sqrt(discriminant)
    ) / (4.0 * product_q)
    b = (a - r) / 2.0
    weights = weights_from_ab(qx, qy, a, b)
    if np.min(weights) <= 1e-12:
        return None

    product_a = 1.0 - qy - (1.0 - qx) * r
    product_b = qy - qx * r
    if not 0.0 < product_a < 1.0 or not 0.0 < product_b < 1.0:
        return None
    entropy_r_slope = -(
        (1.0 - qx) * math.log2((1.0 - product_a) / product_a)
        + qx * math.log2((1.0 - product_b) / product_b)
    )
    a_slope = 4.0 * product_q * r + c * (1.0 - 2.0 * qy)
    return ProductBalancePoint(
        r=float(r),
        b=float(b),
        entropy=float(hb(product_a) + hb(product_b)),
        a_slope=float(a_slope),
        entropy_r_slope=float(entropy_r_slope),
        weights=tuple(float(value) for value in weights),
    )


def rate_active_explicit_certificate(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
) -> bool:
    """Root-free sufficient test for a strict rate-active K=4 optimum.

    Besides the face-entropy window, this checks an algebraic product-balance
    branch.  Positive da/dr at the left endpoint, positive dg/dr at the right
    endpoint, and Rc below the left product-balance entropy imply
    b'(a)<1/2 throughout [a_C,a_R].  Hence the implicit root-shift condition
    follows by integration, without solving either endpoint entropy root.
    """

    if not 0.0 < qy < qx < 0.5:
        return False
    q_s = qx + crossover - 2.0 * qx * crossover
    h_good = hb(crossover)
    h_bad = hb(q_s)
    if h_bad <= h_good + 1e-14:
        return False
    a_c = (h_bad - task_entropy) / (h_bad - h_good)
    a_r = rate / hb(qx)
    a_y = qy / (1.0 - qx)
    a_t = qy / qx
    if not a_y < a_c < a_r < a_t:
        return False

    c = 1.0 - 2.0 * qx

    def b_upper(value: float) -> float:
        return (qy - qx * value) / c

    h_123_c = evaluate(
        qx, qy, crossover, a_c, b_upper(a_c)
    ).common_randomness
    h_123_r = evaluate(
        qx, qy, crossover, a_r, b_upper(a_r)
    ).common_randomness
    h_134_c = evaluate(qx, qy, crossover, a_c, 0.0).common_randomness
    h_134_r = evaluate(qx, qy, crossover, a_r, 0.0).common_randomness
    if not max(h_123_c, h_123_r) < randomness_rate < min(
        h_134_c, h_134_r
    ):
        return False

    product_c = product_balance_at_fixed_a(qx, qy, a_c)
    product_r = product_balance_at_fixed_a(qx, qy, a_r)
    if product_c is None or product_r is None:
        return False
    if not product_c.r < product_r.r:
        return False
    if product_c.a_slope <= 0.0:
        return False
    if product_r.entropy_r_slope <= 0.0:
        return False
    return randomness_rate < product_c.entropy


def post_face_entry_task_left_certificate(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
) -> bool:
    """Sufficient test for a strict task-left optimum after F123 enters.

    This is the endpoint mirror of the implicit rate-active root-shift
    corollary.  In the regime a_Y<a_C<a_R<a_T, the common face-entropy
    window creates unique all-positive roots b_C and b_R at both resource
    endpoints and leaves F123 at a_R as the only K=3 competitor.  The exact
    identity

        Phi(a_R)-Phi(a_C)
          = 4*qx*(1-qx)*(
                b_R-b_C-(a_R-a_C)/2
            )

    shows that b_R-b_C>(a_R-a_C)/2 makes the task endpoint globally best
    among the finite K=4 candidates.  The inequalities are strict, so the
    accepted set is an open sufficient region, not a necessary condition.
    """

    if not 0.0 < qy < qx < 0.5:
        return False
    q_s = qx + crossover - 2.0 * qx * crossover
    h_good = hb(crossover)
    h_bad = hb(q_s)
    if h_bad <= h_good + 1e-14:
        return False
    a_c = (h_bad - task_entropy) / (h_bad - h_good)
    a_r = rate / hb(qx)
    a_y = qy / (1.0 - qx)
    a_t = qy / qx
    if not a_y < a_c < a_r < a_t:
        return False

    c = 1.0 - 2.0 * qx

    def b_upper(value: float) -> float:
        return (qy - qx * value) / c

    h_123_c = evaluate(
        qx, qy, crossover, a_c, b_upper(a_c)
    ).common_randomness
    h_123_r = evaluate(
        qx, qy, crossover, a_r, b_upper(a_r)
    ).common_randomness
    h_134_c = evaluate(
        qx, qy, crossover, a_c, 0.0
    ).common_randomness
    h_134_r = evaluate(
        qx, qy, crossover, a_r, 0.0
    ).common_randomness
    if not max(h_123_c, h_123_r) < randomness_rate < min(
        h_134_c, h_134_r
    ):
        return False

    left = optimize_b_at_fixed_a(
        qx, qy, crossover, a_c, randomness_rate
    )
    right = optimize_b_at_fixed_a(
        qx, qy, crossover, a_r, randomness_rate
    )
    if left is None or right is None:
        return False
    b_c = left.weights[1]
    b_r = right.weights[1]
    return b_r - b_c > (a_r - a_c) / 2.0


def post_face_entry_task_left_explicit_certificate(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
) -> bool:
    """Root-free sufficient test for the post-face-entry task endpoint.

    The face geometry is identical to the implicit task-left certificate.
    On the algebraic product-balance branch, positive da/dr at a_C and
    positive dg/dr at a_R make H_x(a) strictly increasing on [a_C,a_R].
    If Rc lies above H_x(a_R), every selected descending entropy root lies
    to the left of its product-balance point.  Hence b'(a)>1/2 throughout
    the interval and the implicit task-left root-shift follows by
    integration, without solving either endpoint entropy root.
    """

    if not 0.0 < qy < qx < 0.5:
        return False
    q_s = qx + crossover - 2.0 * qx * crossover
    h_good = hb(crossover)
    h_bad = hb(q_s)
    if h_bad <= h_good + 1e-14:
        return False
    a_c = (h_bad - task_entropy) / (h_bad - h_good)
    a_r = rate / hb(qx)
    a_y = qy / (1.0 - qx)
    a_t = qy / qx
    if not a_y < a_c < a_r < a_t:
        return False

    c = 1.0 - 2.0 * qx

    def b_upper(value: float) -> float:
        return (qy - qx * value) / c

    h_123_c = evaluate(
        qx, qy, crossover, a_c, b_upper(a_c)
    ).common_randomness
    h_123_r = evaluate(
        qx, qy, crossover, a_r, b_upper(a_r)
    ).common_randomness
    h_134_c = evaluate(
        qx, qy, crossover, a_c, 0.0
    ).common_randomness
    h_134_r = evaluate(
        qx, qy, crossover, a_r, 0.0
    ).common_randomness
    if not max(h_123_c, h_123_r) < randomness_rate < min(
        h_134_c, h_134_r
    ):
        return False

    product_c = product_balance_at_fixed_a(qx, qy, a_c)
    product_r = product_balance_at_fixed_a(qx, qy, a_r)
    if product_c is None or product_r is None:
        return False
    if not product_c.r < product_r.r:
        return False
    if product_c.a_slope <= 0.0:
        return False
    if product_r.entropy_r_slope <= 0.0:
        return False
    return randomness_rate > product_r.entropy


def task_active_face_transition_certificate(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
) -> bool:
    """Sufficient test for the high-rate task-active K=4 region.

    Once F123 enters, let a_123 be its exact K=3 optimum and compare it with
    the distortion crossing

        a_D = 2*qy + (1-2*qx)*(a_C-2*b_C),

    where b_C is the unique four-mode entropy root at the task endpoint.
    The identity

        D_123(a_123)-D(a_C,b_C)
          = 2*qx*(1-qx)/(1-2*qx) * (a_D-a_123)

    makes a_123<a_D the only new face-dominance scalar condition.  The two
    F134 endpoint entropies exclude that strictly concave face over the full
    rate interval.
    """

    if not 0.0 < qy < qx < 0.5:
        return False
    q_s = qx + crossover - 2.0 * qx * crossover
    h_good = hb(crossover)
    h_bad = hb(q_s)
    if h_bad <= h_good + 1e-14:
        return False
    a_c = (h_bad - task_entropy) / (h_bad - h_good)
    a_y = qy / (1.0 - qx)
    a_r = rate / hb(qx)
    a_t = qy / qx
    if not 0.0 < a_c < a_y < a_r:
        return False

    upper_134 = min(a_t, a_r)
    h_234_c = evaluate(
        qx, qy, crossover, a_c, a_c
    ).common_randomness
    h_134_c = evaluate(
        qx, qy, crossover, a_c, 0.0
    ).common_randomness
    h_134_upper = evaluate(
        qx, qy, crossover, upper_134, 0.0
    ).common_randomness
    if not h_234_c < randomness_rate < min(h_134_c, h_134_upper):
        return False

    task_point = optimize_b_at_fixed_a(
        qx, qy, crossover, a_c, randomness_rate
    )
    if task_point is None:
        return False
    b_c = task_point.weights[1]
    if not 0.0 < b_c < a_c:
        return False
    distortion_crossing = (
        2.0 * qy + (1.0 - 2.0 * qx) * (a_c - 2.0 * b_c)
    )

    face_123 = optimize_face(
        "123",
        qx,
        qy,
        crossover,
        rate,
        task_entropy,
        randomness_rate,
    )
    if face_123 is None:
        return True
    a_123 = face_123.metric.weights[0] + face_123.metric.weights[1]
    return a_123 < distortion_crossing


def task_active_unified_certificate(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
) -> bool:
    """Unified sufficient test for the task-active strict K=4 region.

    The common entropy window produces an all-positive K=4 point at the task
    endpoint a_C and excludes F134 on its complete feasible interval.  If the
    rate endpoint does not exceed a_Y, F123 has not entered and the remaining
    F234 endpoint is strictly worse.  Above a_Y, the only additional check is
    the F123 distortion crossing used by
    :func:`task_active_face_transition_certificate`.

    This routine is deliberately only a sufficient certificate; it does not
    assert that the stated conditions are necessary outside the audited
    Bernoulli slice.
    """

    if not 0.0 < qy < qx < 0.5:
        return False
    q_s = qx + crossover - 2.0 * qx * crossover
    h_good = hb(crossover)
    h_bad = hb(q_s)
    if h_bad <= h_good + 1e-14:
        return False
    a_c = (h_bad - task_entropy) / (h_bad - h_good)
    a_y = qy / (1.0 - qx)
    a_r = rate / hb(qx)
    a_t = qy / qx
    if not 0.0 < a_c < a_y or not a_c < a_r:
        return False

    upper_134 = min(a_t, a_r)
    h_234_c = evaluate(
        qx, qy, crossover, a_c, a_c
    ).common_randomness
    h_134_c = evaluate(
        qx, qy, crossover, a_c, 0.0
    ).common_randomness
    h_134_upper = evaluate(
        qx, qy, crossover, upper_134, 0.0
    ).common_randomness
    if not h_234_c < randomness_rate < min(h_134_c, h_134_upper):
        return False

    task_point = optimize_b_at_fixed_a(
        qx, qy, crossover, a_c, randomness_rate
    )
    if task_point is None:
        return False
    b_c = task_point.weights[1]
    if not 0.0 < b_c < a_c:
        return False

    # At a_R=a_Y, F123 only touches F234 at their common two-mode endpoint,
    # so it cannot beat the all-positive task point either.
    if a_r <= a_y + 1e-12:
        return True

    distortion_crossing = (
        2.0 * qy + (1.0 - 2.0 * qx) * (a_c - 2.0 * b_c)
    )
    face_123 = optimize_face(
        "123",
        qx,
        qy,
        crossover,
        rate,
        task_entropy,
        randomness_rate,
    )
    if face_123 is None:
        return True
    a_123 = face_123.metric.weights[0] + face_123.metric.weights[1]
    return a_123 < distortion_crossing


def grid_search(
    qx: float,
    qy: float,
    crossover: float,
    rate: float,
    task_entropy: float,
    randomness_rate: float,
    max_modes: int,
    grid: int = 1001,
    task_mode_aware: bool = True,
) -> Metrics | None:
    best: Metrics | None = None
    tolerance = 2e-9
    for a in np.linspace(0.0, 1.0, grid):
        if a * hb(qx) > rate + tolerance:
            continue
        for b in np.linspace(0.0, a, max(2, int(grid * a) + 1)):
            weights = weights_from_ab(qx, qy, float(a), float(b))
            if np.min(weights) < -tolerance:
                continue
            weights[np.abs(weights) < tolerance] = 0.0
            metric = evaluate(qx, qy, crossover, float(a), float(b))
            task_value = (
                metric.task_entropy_mode_aware
                if task_mode_aware
                else metric.task_entropy_output_only
            )
            if task_value > task_entropy + tolerance:
                continue
            if metric.common_randomness > randomness_rate + tolerance:
                continue
            if metric.support > max_modes:
                continue
            if best is None or metric.distortion < best.distortion - 1e-12:
                best = metric
    return best


def pretty(metric: Metrics | None) -> str:
    if metric is None:
        return "infeasible"
    return (
        f"D={metric.distortion:.6f}, R={metric.payload_rate:.6f}, "
        f"Rc={metric.common_randomness:.6f}, K={metric.support}, "
        f"H(S|Y,V)={metric.task_entropy_mode_aware:.6f}, "
        f"H(S|Y)={metric.task_entropy_output_only:.6f}, "
        f"w={tuple(round(x, 6) for x in metric.weights)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=401)
    args = parser.parse_args()

    qx, qy, crossover = 0.3, 0.25, 0.2
    h_x = hb(qx)
    loose_c = 1.0

    print("Audit: output-only versus mode-aware task entropy")
    audit = evaluate(qx, qy, crossover, 0.5, 0.0)
    print(pretty(audit))
    print()

    print("Corrected unlimited-randomness candidate and grid check")
    for rate in (0.2, 0.5, 0.75, h_x):
        exact = closed_form_unlimited_mode_aware(qx, qy, crossover, rate, loose_c)
        numeric = grid_search(
            qx,
            qy,
            crossover,
            rate,
            loose_c,
            randomness_rate=10.0,
            max_modes=4,
            grid=args.grid,
        )
        print(f"rate budget={rate:.6f}")
        print("  closed:", pretty(exact))
        print("  grid:  ", pretty(numeric))
    print()

    print("Finite-mode / limited-common-randomness examples")
    rate = 0.75
    for k in (1, 2, 3, 4):
        for rc in (0.0, 0.7, 1.0, 1.5, 2.0):
            numeric = grid_search(
                qx,
                qy,
                crossover,
                rate,
                loose_c,
                randomness_rate=rc,
                max_modes=k,
                grid=args.grid,
            )
            print(f"K={k}, Rc={rc:.2f}: {pretty(numeric)}")

    print()
    print("Strict K=3 versus K=4 witness")
    witness = dict(
        qx=0.18,
        qy=0.10,
        crossover=0.235,
        rate=0.082,
        task_entropy=0.90,
        randomness_rate=0.78,
    )
    k2 = optimize_k2(**witness)
    k3 = optimize_k3(**witness)
    k4 = optimize_k4_finite_candidates(**witness)
    k4_grid = optimize_k4_numeric(**witness)
    print("  K=2:", pretty(None if k2 is None else k2.metric))
    print("  K=3:", pretty(None if k3 is None else k3.metric))
    print("  K=4 finite:", pretty(None if k4 is None else k4.metric))
    print("  K=4 grid:  ", pretty(None if k4_grid is None else k4_grid.metric))


if __name__ == "__main__":
    main()
