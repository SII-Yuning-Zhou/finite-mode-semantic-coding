"""Regression checks for the common M/O frontier and its exclusion witness.

The numerical checks complement the manuscript proof. The model-O benchmark
is obtained independently on its flip-free scalar branch, then compared with
the existing exact model-M face/endpoint solvers.
"""
import math
import unittest

from theory.bernoulli_mode_limited import (
    hb, optimize_k3, optimize_k4_finite_candidates,
)


def output_task(q, y, epsilon, r):
    s = q + epsilon - 2 * q * epsilon
    kappa = q * (1 - q) * (1 - 2 * epsilon)
    return y * hb(s + kappa * r / y) + (1 - y) * hb(
        s - kappa * r / (1 - y)
    )


def branch_entropy(q, y, r):
    weights = (r, 1 - y - (1 - q) * r, y - q * r)
    return -sum(p * math.log2(p) for p in weights if p > 0)


class ReceiverVisibilityFeasibility(unittest.TestCase):
    def test_strict_four_mode_witness_excludes_both_output_branches(self):
        q, y, epsilon = 0.1, 0.022, 0.1
        task = 0.98 * hb(0.18) + 0.02 * hb(epsilon)
        rate = 0.021 * hb(q)
        k3 = optimize_k3(q, y, epsilon, rate, task, 0.26)
        k4 = optimize_k4_finite_candidates(q, y, epsilon, rate, task, 0.26)
        self.assertIsNotNone(k3)
        self.assertIsNotNone(k4)
        self.assertLess(k4.metric.distortion, k3.metric.distortion)
        self.assertEqual(k4.metric.support, 4)
        for sign in (-1, 1):
            # The rate budget bounds |r| even before randomness is charged.
            self.assertGreater(output_task(q, y, epsilon, sign * 0.021), task)
        self.assertAlmostEqual(output_task(q, y, epsilon, 0.021), 0.679599811457608)
        self.assertAlmostEqual(output_task(q, y, epsilon, -0.021), 0.679493989875197)

    def test_common_global_frontier_from_feasible_output_only_schedules(self):
        cases = 0
        for q in (0.1, 0.18, 0.35):
            for y in (0.022, 0.1, 0.4):
                for epsilon in (0.1, 0.235):
                    for fraction in (0.35, 0.85):
                        rmax = min(y / q, (1 - y) / (1 - q))
                        r0 = fraction * rmax
                        ru = r0 + 0.25 * (rmax - r0)
                        rate = ru * hb(q)
                        task = output_task(q, y, epsilon, r0) + 1e-9
                        rc = branch_entropy(q, y, r0) + 0.03
                        # r0 is an explicit three-mode O-feasible certificate.
                        if branch_entropy(q, y, ru) <= rc:
                            ro = ru
                        else:
                            lo, hi = r0, ru
                            for _ in range(70):
                                mid = (lo + hi) / 2
                                if branch_entropy(q, y, mid) <= rc:
                                    lo = mid
                                else:
                                    hi = mid
                            ro = lo
                        do = q + y - 2 * q * y - 2 * q * (1 - q) * ro
                        with self.subTest(q=q, y=y, epsilon=epsilon, fraction=fraction):
                            for solver in (optimize_k3, optimize_k4_finite_candidates):
                                point = solver(q, y, epsilon, rate, task, rc)
                                self.assertIsNotNone(point)
                                self.assertAlmostEqual(point.metric.distortion, do, delta=2e-8)
                        cases += 1
        self.assertEqual(cases, 36)


if __name__ == "__main__":
    unittest.main()
