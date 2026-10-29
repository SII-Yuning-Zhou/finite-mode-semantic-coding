"""Interval certificate for a strict K=4 versus K=3 open-set witness.

The script uses outward-rounded mpmath interval arithmetic.  It certifies the
strict inequalities in the sufficient-condition corollary and brackets the
unique four-mode entropy root without relying on binary floating-point point
comparisons.
"""

from __future__ import annotations

import mpmath as mp


mp.iv.dps = 80
iv = mp.iv


def point(decimal: str):
    return iv.mpf([decimal, decimal])


def hb(value):
    return -(
        value * iv.log(value, 2)
        + (1 - value) * iv.log(1 - value, 2)
    )


def entropy(positive_weights):
    return sum(-weight * iv.log(weight, 2) for weight in positive_weights)


def main() -> None:
    q = point("0.18")
    y = point("0.10")
    crossover = point("0.235")
    task_budget = point("0.90")
    rate_budget = point("0.082")
    randomness_budget = point("0.78")

    q_s = q + crossover - 2 * q * crossover
    h_q = hb(q)
    h_noise = hb(crossover)
    h_s = hb(q_s)
    a_c = (h_s - task_budget) / (h_s - h_noise)
    a_y = y / (1 - q)

    # Open-set geometry: 0<y<q<1/2, active task lower endpoint, and the rate
    # upper endpoint strictly between a_c and the first one-absent face.
    assert point("0") < y < q < point("0.5")
    assert point("0") < a_c < a_y
    rate_lower = a_c * h_q
    rate_upper = a_y * h_q
    assert rate_lower < rate_budget < rate_upper

    # Flip-absent face F134: (identity, zero, one).
    def h_134(a):
        return entropy(
            (
                a,
                1 - y - (1 - q) * a,
                y - q * a,
            )
        )

    # Identity-absent face F234: (flip, zero, one).
    def h_234(a):
        return entropy(
            (
                a,
                1 - y - q * a,
                y - (1 - q) * a,
            )
        )

    h_234_ac = h_234(a_c)
    h_134_ac = h_134(a_c)
    h_134_ay = h_134(a_y)
    assert h_234_ac < randomness_budget
    assert randomness_budget < h_134_ac
    assert randomness_budget < h_134_ay

    # Bracket the unique fixed-a_c descending entropy root.  The strict
    # endpoint inequalities above prove uniqueness; these rational decimal
    # endpoints certify its numerical location.
    root_lower = point("0.08703231494280")
    root_upper = point("0.08703231494281")
    c = 1 - 2 * q

    def h_four(b):
        return entropy(
            (
                a_c - b,
                b,
                1 - y - (1 - q) * a_c + c * b,
                y - q * a_c - c * b,
            )
        )

    assert point("0") < root_lower < root_upper < a_c
    assert h_four(root_lower) > randomness_budget
    assert h_four(root_upper) < randomness_budget

    # D3-D4=4q(1-q)(a_c-b*).  Since b*<=root_upper, the following is a
    # rigorous lower bound on the strict distortion gap.
    gap_lower = 4 * q * (1 - q) * (a_c - root_upper)
    assert gap_lower > point("0.01895")

    print("Strict K=4 open-set interval certificate: PASS")
    print(f"  a_C       = {a_c}")
    print(f"  a_Y       = {a_y}")
    print(f"  rate low  = {rate_lower}")
    print(f"  rate high = {rate_upper}")
    print(f"  H234(a_C) = {h_234_ac}")
    print(f"  H134(a_C) = {h_134_ac}")
    print(f"  H134(a_Y) = {h_134_ay}")
    print(f"  b* bracket= [{root_lower}, {root_upper}]")
    print(f"  gap lower = {gap_lower}")


if __name__ == "__main__":
    main()

