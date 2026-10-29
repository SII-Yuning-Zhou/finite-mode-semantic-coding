"""Certified active-set enumeration for a non-Bernoulli finite-alphabet example.

The example has a ternary source X, a binary semantic state S, and a binary
target reconstruction Y.  With unrestricted schedule entropy the optimization
is a linear program in the deterministic-map weights.  Since |Y|=2 and there
are two affine resource inequalities, every LP vertex has at most four active
maps.  The script enumerates every possible active set, encloses all entropy
coefficients and vertex solutions with mpmath interval arithmetic, and writes
the complete certificate table.

This is a numerical certificate for the stated canonical example.  It does not
replace the analytic support theorem and does not claim to solve the finite-Rc
nonconvex program for arbitrary alphabets.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import mpmath as mp


MP_DPS = 80
mp.mp.dps = MP_DPS
mp.iv.dps = MP_DPS


P_X = (Fraction(35, 100), Fraction(41, 100), Fraction(24, 100))
P_S1_GIVEN_X = (Fraction(18, 100), Fraction(30, 100), Fraction(72, 100))
X_LEVELS = (Fraction(0, 1), Fraction(1, 2), Fraction(1, 1))
Q_Y1 = Fraction(68, 100)
RATE_BUDGET = Fraction(57, 100)
TASK_BUDGET = Fraction(86, 100)
COMMON_RANDOMNESS_BUDGET = Fraction(19, 10)


def mpf(value: Fraction | int) -> mp.mpf:
    value = Fraction(value)
    return mp.mpf(value.numerator) / value.denominator


def ivf(value: Fraction | int):
    value = Fraction(value)
    return mp.iv.mpf(value.numerator) / value.denominator


def binary_entropy(value: mp.mpf) -> mp.mpf:
    if value == 0 or value == 1:
        return mp.mpf(0)
    return -(value * mp.log(value, 2) + (1 - value) * mp.log(1 - value, 2))


def binary_entropy_iv(value, exact_zero: bool = False, exact_one: bool = False):
    if exact_zero or exact_one:
        return mp.iv.mpf(0)
    return -(
        value * mp.iv.log(value) + (1 - value) * mp.iv.log(1 - value)
    ) / mp.iv.log(2)


def interval_bounds(value) -> tuple[float, float]:
    return float(value.a), float(value.b)


def interval_string(value, digits: int = 55) -> str:
    return mp.nstr(value, digits)


@dataclass(frozen=True)
class MapCoefficient:
    index: int
    mapping: tuple[int, int, int]
    target: mp.mpf
    rate: mp.mpf
    task: mp.mpf
    distortion: mp.mpf
    target_iv: object
    rate_iv: object
    task_iv: object
    distortion_iv: object


@dataclass
class Vertex:
    support: tuple[int, ...]
    active_resources: tuple[str, ...]
    weights: list[mp.mpf]
    weights_iv: list[object]
    rate: mp.mpf
    rate_iv: object
    task: mp.mpf
    task_iv: object
    distortion: mp.mpf
    distortion_iv: object

    @property
    def support_size(self) -> int:
        return len(self.support)


def build_coefficients() -> list[MapCoefficient]:
    coefficients: list[MapCoefficient] = []
    for index, mapping in enumerate(itertools.product((0, 1), repeat=3)):
        target_fraction = sum(
            P_X[x] * mapping[x] for x in range(len(P_X))
        )
        distortion_fraction = sum(
            P_X[x] * (X_LEVELS[x] - mapping[x]) ** 2
            for x in range(len(P_X))
        )

        target = mpf(target_fraction)
        target_iv = ivf(target_fraction)
        rate = binary_entropy(target)
        rate_iv = binary_entropy_iv(
            target_iv,
            exact_zero=target_fraction == 0,
            exact_one=target_fraction == 1,
        )

        task = mp.mpf(0)
        task_iv = mp.iv.mpf(0)
        for y in (0, 1):
            p_y = sum(P_X[x] for x in range(3) if mapping[x] == y)
            p_s1_y = sum(
                P_X[x] * P_S1_GIVEN_X[x]
                for x in range(3)
                if mapping[x] == y
            )
            if p_y == 0:
                continue
            posterior = p_s1_y / p_y
            task += mpf(p_y) * binary_entropy(mpf(posterior))
            task_iv += ivf(p_y) * binary_entropy_iv(
                ivf(posterior),
                exact_zero=posterior == 0,
                exact_one=posterior == 1,
            )

        coefficients.append(
            MapCoefficient(
                index=index,
                mapping=tuple(mapping),
                target=target,
                rate=rate,
                task=task,
                distortion=mpf(distortion_fraction),
                target_iv=target_iv,
                rate_iv=rate_iv,
                task_iv=task_iv,
                distortion_iv=ivf(distortion_fraction),
            )
        )
    return coefficients


def dot_point(values: Sequence[mp.mpf], weights: Sequence[mp.mpf]) -> mp.mpf:
    return sum(value * weight for value, weight in zip(values, weights))


def dot_interval(values: Sequence[object], weights: Sequence[object]):
    return sum(
        (value * weight for value, weight in zip(values, weights)),
        mp.iv.mpf(0),
    )


def enumerate_vertices(
    coefficients: Sequence[MapCoefficient],
) -> tuple[list[Vertex], dict[str, int]]:
    vertices: list[Vertex] = []
    counts = {
        "total_vertex_systems": 0,
        "support_size_2_systems": 0,
        "support_size_3_systems": 0,
        "support_size_4_systems": 0,
        "singular": 0,
        "certified_negative_weight": 0,
        "certified_budget_violation": 0,
        "certified_feasible": 0,
        "ambiguous": 0,
    }

    rate_budget = mpf(RATE_BUDGET)
    task_budget = mpf(TASK_BUDGET)
    rate_budget_iv = ivf(RATE_BUDGET)
    task_budget_iv = ivf(TASK_BUDGET)

    # No single map has target mass Q_Y1 in this example.  For k >= 2,
    # normalization and target equality supply two active rows.  A k-map LP
    # vertex therefore needs k-2 independent active resource inequalities.
    for support_size in range(2, 5):
        for support in itertools.combinations(range(len(coefficients)), support_size):
            for active_resources in itertools.combinations(
                ("rate", "task"), support_size - 2
            ):
                counts["total_vertex_systems"] += 1
                counts[f"support_size_{support_size}_systems"] += 1
                rows = [
                    [mp.mpf(1) for _ in support],
                    [coefficients[j].target for j in support],
                ]
                rhs = [mp.mpf(1), mpf(Q_Y1)]
                rows_iv = [
                    [mp.iv.mpf(1) for _ in support],
                    [coefficients[j].target_iv for j in support],
                ]
                rhs_iv = [mp.iv.mpf(1), ivf(Q_Y1)]

                if "rate" in active_resources:
                    rows.append([coefficients[j].rate for j in support])
                    rhs.append(rate_budget)
                    rows_iv.append([coefficients[j].rate_iv for j in support])
                    rhs_iv.append(rate_budget_iv)
                if "task" in active_resources:
                    rows.append([coefficients[j].task for j in support])
                    rhs.append(task_budget)
                    rows_iv.append([coefficients[j].task_iv for j in support])
                    rhs_iv.append(task_budget_iv)

                matrix = mp.matrix(rows)
                determinant = mp.det(matrix)
                if abs(determinant) < mp.mpf("1e-65"):
                    counts["singular"] += 1
                    continue

                try:
                    weights = list(mp.lu_solve(matrix, mp.matrix(rhs)))
                    weights_iv = list(
                        mp.iv.lu_solve(mp.iv.matrix(rows_iv), mp.iv.matrix(rhs_iv))
                    )
                except (ZeroDivisionError, ValueError):
                    counts["ambiguous"] += 1
                    continue

                weight_bounds = [interval_bounds(weight) for weight in weights_iv]
                if any(upper < 0 for _, upper in weight_bounds):
                    counts["certified_negative_weight"] += 1
                    continue

                rate_values = [coefficients[j].rate for j in support]
                task_values = [coefficients[j].task for j in support]
                distortion_values = [coefficients[j].distortion for j in support]
                rate_values_iv = [coefficients[j].rate_iv for j in support]
                task_values_iv = [coefficients[j].task_iv for j in support]
                distortion_values_iv = [
                    coefficients[j].distortion_iv for j in support
                ]

                rate = dot_point(rate_values, weights)
                task = dot_point(task_values, weights)
                distortion = dot_point(distortion_values, weights)
                rate_iv = dot_interval(rate_values_iv, weights_iv)
                task_iv = dot_interval(task_values_iv, weights_iv)
                distortion_iv = dot_interval(distortion_values_iv, weights_iv)

                rate_lower, rate_upper = interval_bounds(rate_iv)
                task_lower, task_upper = interval_bounds(task_iv)
                if rate_lower > float(rate_budget) or task_lower > float(task_budget):
                    counts["certified_budget_violation"] += 1
                    continue

                definitely_nonnegative = all(lower >= 0 for lower, _ in weight_bounds)
                definitely_feasible = (
                    rate_upper <= float(rate_budget)
                    or "rate" in active_resources
                ) and (
                    task_upper <= float(task_budget)
                    or "task" in active_resources
                )
                if not definitely_nonnegative or not definitely_feasible:
                    counts["ambiguous"] += 1
                    continue

                counts["certified_feasible"] += 1
                vertices.append(
                    Vertex(
                        support=support,
                        active_resources=active_resources,
                        weights=weights,
                        weights_iv=weights_iv,
                        rate=rate,
                        rate_iv=rate_iv,
                        task=task,
                        task_iv=task_iv,
                        distortion=distortion,
                        distortion_iv=distortion_iv,
                    )
                )

    vertices.sort(key=lambda vertex: vertex.distortion)
    return vertices, counts


def schedule_entropy(weights: Iterable[mp.mpf]) -> mp.mpf:
    return -sum(weight * mp.log(weight, 2) for weight in weights if weight > 0)


def schedule_entropy_iv(weights: Iterable[object]):
    return -sum(
        (
            weight * mp.iv.log(weight) / mp.iv.log(2)
            for weight in weights
            if interval_bounds(weight)[1] > 0
        ),
        mp.iv.mpf(0),
    )


def vertex_record(vertex: Vertex, coefficients: Sequence[MapCoefficient]) -> dict:
    return {
        "support_size": vertex.support_size,
        "support_indices": list(vertex.support),
        "support_maps": ["".join(map(str, coefficients[j].mapping)) for j in vertex.support],
        "active_resources": list(vertex.active_resources),
        "weights": [mp.nstr(weight, 40) for weight in vertex.weights],
        "weight_intervals": [interval_string(weight) for weight in vertex.weights_iv],
        "rate": mp.nstr(vertex.rate, 40),
        "rate_interval": interval_string(vertex.rate_iv),
        "task": mp.nstr(vertex.task, 40),
        "task_interval": interval_string(vertex.task_iv),
        "distortion": mp.nstr(vertex.distortion, 40),
        "distortion_interval": interval_string(vertex.distortion_iv),
    }


def write_outputs(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficients = build_coefficients()
    vertices, counts = enumerate_vertices(coefficients)
    if counts["ambiguous"]:
        raise RuntimeError(
            f"Interval audit left {counts['ambiguous']} LP vertex systems ambiguous."
        )
    if not vertices:
        raise RuntimeError("No certified feasible LP vertices were found.")

    best_k4 = min(
        (vertex for vertex in vertices if vertex.support_size <= 4),
        key=lambda vertex: vertex.distortion,
    )
    best_k3 = min(
        (vertex for vertex in vertices if vertex.support_size <= 3),
        key=lambda vertex: vertex.distortion,
    )
    gap_iv = best_k3.distortion_iv - best_k4.distortion_iv
    gap_lower, _ = interval_bounds(gap_iv)
    if gap_lower <= 0:
        raise RuntimeError(f"Strict K=4 gap was not certified: {gap_iv}")

    entropy_k4 = schedule_entropy(best_k4.weights)
    entropy_k4_iv = schedule_entropy_iv(best_k4.weights_iv)
    _, entropy_upper = interval_bounds(entropy_k4_iv)
    if entropy_upper >= float(mpf(COMMON_RANDOMNESS_BUDGET)):
        raise RuntimeError(
            "The four-mode optimum does not meet the declared common-randomness budget."
        )
    entropy_k3 = schedule_entropy(best_k3.weights)
    entropy_k3_iv = schedule_entropy_iv(best_k3.weights_iv)
    _, entropy_k3_upper = interval_bounds(entropy_k3_iv)
    if entropy_k3_upper >= float(mpf(COMMON_RANDOMNESS_BUDGET)):
        raise RuntimeError(
            "The three-mode optimum does not meet the declared common-randomness budget."
        )

    map_rows = []
    for coefficient in coefficients:
        map_rows.append(
            {
                "index": coefficient.index,
                "mapping": "".join(map(str, coefficient.mapping)),
                "target_y1": mp.nstr(coefficient.target, 40),
                "payload_entropy": mp.nstr(coefficient.rate, 40),
                "task_entropy": mp.nstr(coefficient.task, 40),
                "distortion": mp.nstr(coefficient.distortion, 40),
            }
        )

    summary = {
        "certificate": "ternary_source_binary_target_free_entropy_lp_v1",
        "precision_decimal_digits": MP_DPS,
        "source_distribution": [str(value) for value in P_X],
        "semantic_channel_p_s1_given_x": [str(value) for value in P_S1_GIVEN_X],
        "source_levels": [str(value) for value in X_LEVELS],
        "target_p_y1": str(Q_Y1),
        "payload_budget": str(RATE_BUDGET),
        "task_budget": str(TASK_BUDGET),
        "common_randomness_budget": str(COMMON_RANDOMNESS_BUDGET),
        "vertex_system_counts": counts,
        "map_coefficients": map_rows,
        "best_k4": vertex_record(best_k4, coefficients),
        "best_k3": vertex_record(best_k3, coefficients),
        "strict_distortion_gap_interval": interval_string(gap_iv),
        "best_k4_schedule_entropy": mp.nstr(entropy_k4, 40),
        "best_k4_schedule_entropy_interval": interval_string(entropy_k4_iv),
        "best_k3_schedule_entropy": mp.nstr(entropy_k3, 40),
        "best_k3_schedule_entropy_interval": interval_string(entropy_k3_iv),
        "claims": {
            "all_vertex_systems_classified": counts["ambiguous"] == 0,
            "strict_k4_over_k3": gap_lower > 0,
            "k4_feasible_at_rc_1p9": entropy_upper
            < float(mpf(COMMON_RANDOMNESS_BUDGET)),
            "k3_feasible_at_rc_1p9": entropy_k3_upper
            < float(mpf(COMMON_RANDOMNESS_BUDGET)),
            "free_entropy_support_bound_m_plus_2_is_sharp_for_this_example": best_k4.support_size
            == 4,
        },
    }

    summary_path = output_dir / "ternary_binary_certificate_v1.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    vertices_path = output_dir / "ternary_binary_vertices_v1.csv"
    with vertices_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "support_size",
                "support_indices",
                "support_maps",
                "active_resources",
                "distortion",
                "distortion_interval",
                "rate",
                "task",
                "weights",
            ],
        )
        writer.writeheader()
        for rank, vertex in enumerate(vertices, start=1):
            record = vertex_record(vertex, coefficients)
            writer.writerow(
                {
                    "rank": rank,
                    "support_size": record["support_size"],
                    "support_indices": ";".join(map(str, record["support_indices"])),
                    "support_maps": ";".join(record["support_maps"]),
                    "active_resources": ";".join(record["active_resources"]),
                    "distortion": record["distortion"],
                    "distortion_interval": record["distortion_interval"],
                    "rate": record["rate"],
                    "task": record["task"],
                    "weights": ";".join(record["weights"]),
                }
            )

    return {
        "summary_path": str(summary_path),
        "vertices_path": str(vertices_path),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/general_finite_alphabet"),
    )
    args = parser.parse_args()
    result = write_outputs(args.output_dir)
    summary = result["summary"]
    print(f"PASS: {result['summary_path']}")
    print(f"PASS: {result['vertices_path']}")
    print(
        "K4 distortion:",
        summary["best_k4"]["distortion_interval"],
    )
    print(
        "K3 distortion:",
        summary["best_k3"]["distortion_interval"],
    )
    print("Strict gap:", summary["strict_distortion_gap_interval"])
    print(
        "K4 schedule entropy:",
        summary["best_k4_schedule_entropy_interval"],
    )
    print(
        "K3 schedule entropy:",
        summary["best_k3_schedule_entropy_interval"],
    )
    print("LP vertex-system counts:", summary["vertex_system_counts"])


if __name__ == "__main__":
    main()
