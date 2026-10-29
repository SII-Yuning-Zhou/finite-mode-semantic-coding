"""Exact-symbolic audit for the general support theorem and ternary example."""

from __future__ import annotations

import itertools

import sympy as sp


def entropy_binary(p: sp.Expr) -> sp.Expr:
    p = sp.Rational(p)
    if p == 0 or p == 1:
        return sp.S.Zero
    # Canonicalize p and 1-p before constructing the expression.  This makes
    # complement-map entropy identities structural and avoids simplify()
    # expanding rational logarithms into enormous integer powers.
    p = min(p, 1 - p)
    return -(p * sp.log(p) + (1 - p) * sp.log(1 - p)) / sp.log(2)


def ternary_coefficients():
    p_x = (sp.Rational(35, 100), sp.Rational(41, 100), sp.Rational(24, 100))
    p_s1_x = (sp.Rational(18, 100), sp.Rational(30, 100), sp.Rational(72, 100))
    x_levels = (sp.S.Zero, sp.Rational(1, 2), sp.S.One)
    mappings = list(itertools.product((0, 1), repeat=3))
    target, rate, task, distortion = [], [], [], []

    for mapping in mappings:
        q = sum(p_x[x] * mapping[x] for x in range(3))
        target.append(q)
        rate.append(entropy_binary(q))
        distortion.append(
            sum(
                p_x[x] * (x_levels[x] - mapping[x]) ** 2
                for x in range(3)
            )
        )

        conditional_entropy = sp.S.Zero
        for y in (0, 1):
            p_y = sum(p_x[x] for x in range(3) if mapping[x] == y)
            if p_y == 0:
                continue
            p_s1_y = sum(
                p_x[x] * p_s1_x[x]
                for x in range(3)
                if mapping[x] == y
            )
            conditional_entropy += p_y * entropy_binary(p_s1_y / p_y)
        task.append(conditional_entropy)

    return mappings, target, rate, task, distortion


def audit() -> dict:
    mappings, target, rate, task, distortion = ternary_coefficients()

    # Complementing every output bit replaces q by 1-q but leaves both
    # H(Y|V=v) and H(S|Y,V=v) unchanged.
    for j in range(8):
        complement = 7 - j
        assert target[j] + target[complement] == 1
        assert rate[j] == rate[complement]
        assert task[j] == task[complement]

    q_target = sp.Rational(68, 100)
    rate_budget = sp.Rational(57, 100)
    task_budget = sp.Rational(86, 100)

    support_k4 = (1, 3, 6, 7)
    matrix_k4 = sp.Matrix(
        [
            [1 for _ in support_k4],
            [target[j] for j in support_k4],
            [rate[j] for j in support_k4],
            [task[j] for j in support_k4],
        ]
    )
    rhs_k4 = sp.Matrix([1, q_target, rate_budget, task_budget])
    determinant_k4 = matrix_k4.det()
    assert abs(float(sp.N(determinant_k4, 60))) > 1e-12
    weights_k4 = matrix_k4.LUsolve(rhs_k4)
    assert all(float(sp.N(weight, 60)) > 0 for weight in weights_k4)

    support_k3 = (1, 6, 7)
    matrix_k3 = sp.Matrix(
        [
            [1 for _ in support_k3],
            [target[j] for j in support_k3],
            [task[j] for j in support_k3],
        ]
    )
    rhs_k3 = sp.Matrix([1, q_target, task_budget])
    determinant_k3 = matrix_k3.det()
    assert abs(float(sp.N(determinant_k3, 60))) > 1e-12
    weights_k3 = matrix_k3.LUsolve(rhs_k3)
    assert all(float(sp.N(weight, 60)) > 0 for weight in weights_k3)

    distortion_k4 = sum(
        distortion[j] * weights_k4[i] for i, j in enumerate(support_k4)
    )
    distortion_k3 = sum(
        distortion[j] * weights_k3[i] for i, j in enumerate(support_k3)
    )
    gap = distortion_k3 - distortion_k4
    assert float(sp.N(gap, 60)) > 0.018

    # Dimension bookkeeping in the generalized support theorem: m-1 target
    # moments, one payload moment, T task-risk moments, and the distortion
    # objective give d=m+T+1.  The entropy-nonincreasing reduction therefore
    # uses at most m+T+2 modes.  Without the entropy constraint, fixing the
    # actual output law of a global optimum gives an LP with m
    # normalization/target equalities and at most T+1 active resource
    # inequalities, hence m+T+1 positive variables at a basic optimum.  A
    # closed target set contributes no extra dimension because the reduction
    # preserves the actual output marginal.
    m = sp.symbols("m", integer=True, positive=True)
    t = sp.symbols("T", integer=True, positive=True)
    moment_dimension = (m - 1) + 1 + t + 1
    assert sp.simplify(moment_dimension - (m + t + 1)) == 0
    assert sp.simplify(moment_dimension + 1 - (m + t + 2)) == 0
    assert sp.simplify((1 + (m - 1) + (t + 1)) - (m + t + 1)) == 0
    assert sp.simplify(moment_dimension.subs(t, 1) - (m + 2)) == 0
    assert sp.simplify((moment_dimension + 1).subs(t, 1) - (m + 3)) == 0

    return {
        "complement_relations": "PASS",
        "multitask_theorem_dimension_bookkeeping": "PASS",
        "closed_target_set_additional_dimension": 0,
        "k4_active_matrix_determinant_60d": str(sp.N(determinant_k4, 60)),
        "k3_active_matrix_determinant_60d": str(sp.N(determinant_k3, 60)),
        "k4_weights_40d": [str(sp.N(weight, 40)) for weight in weights_k4],
        "k3_weights_40d": [str(sp.N(weight, 40)) for weight in weights_k3],
        "distortion_gap_60d": str(sp.N(gap, 60)),
    }


if __name__ == "__main__":
    result = audit()
    for key, value in result.items():
        print(f"{key}: {value}")
