"""Regression tests pinning the corrected strict-K4-open-set corollary.

Two witnesses from the 2026-09-15 reliability review:

1. Counterexample to the *old* over-strong wording of ``cor:k4-open``.
   With (q, y, eps) = (0.42, 0.25, 0.1), a_C = 0.405, a_R = 0.415,
   Rc = 1.24, every strict hypothesis of the corollary holds and the
   all-positive task endpoint (a_C, b_C) strictly beats the three-mode
   optimum -- yet the payload endpoint (a_R, b_R) is strictly better, so
   the task endpoint is NOT the global four-mode optimum.  The corrected
   statement (feasible witness + guaranteed improvement, global optimum
   decided by the finite-candidate theorem) must keep passing.

2. Positive-correlation model-M existence witness used in
   ``receiver_comparison.tex``: (q, y, eps) = (0.1, 0.022, 0.1),
   a_C = 0.02, a_R = 0.021, Rc = 0.26.  Both endpoint entropy roots lie in
   0 < b < a/2, hence r = a - 2b > 0, and the certified four-mode optimum
   is all-positive with a strict gain.
"""

from __future__ import annotations

import unittest

from theory.bernoulli_mode_limited import (
    evaluate,
    hb,
    optimize_b_at_fixed_a,
    optimize_k3,
    optimize_k4_finite_candidates,
)


def _budgets(q: float, eps: float, a_c: float, a_r: float) -> tuple[float, float]:
    q_s = q + eps - 2.0 * q * eps
    h_s = hb(q_s)
    task = h_s - a_c * (h_s - hb(eps))
    rate = a_r * hb(q)
    return rate, task


class TaskEndpointNotGlobalCounterexample(unittest.TestCase):
    """q=0.42, y=0.25, eps=0.1, a_C=0.405, a_R=0.415, Rc=1.24."""

    Q, Y, EPS = 0.42, 0.25, 0.1
    A_C, A_R, RC = 0.405, 0.415, 1.24

    def setUp(self) -> None:
        self.rate, self.task = _budgets(self.Q, self.EPS, self.A_C, self.A_R)
        self.assertAlmostEqual(self.task, 0.7778918444059833089, places=15)
        self.assertAlmostEqual(self.rate, 0.4073033664389662209, places=15)
        self.k3 = optimize_k3(
            self.Q, self.Y, self.EPS, self.rate, self.task, self.RC
        )
        self.task_point = optimize_b_at_fixed_a(
            self.Q, self.Y, self.EPS, self.A_C, self.RC
        )
        self.payload_point = optimize_b_at_fixed_a(
            self.Q, self.Y, self.EPS, self.A_R, self.RC
        )
        self.k4 = optimize_k4_finite_candidates(
            self.Q, self.Y, self.EPS, self.rate, self.task, self.RC
        )
        for item in (self.k3, self.task_point, self.payload_point, self.k4):
            self.assertIsNotNone(item)

    def test_corollary_hypotheses_all_hold(self) -> None:
        h = lambda a, b: evaluate(  # noqa: E731
            self.Q, self.Y, self.EPS, a, b
        ).common_randomness
        a_y = self.Y / (1.0 - self.Q)
        # (eq:open1) 0 < a_C < a_Y
        self.assertLess(0.0, self.A_C)
        self.assertLess(self.A_C, a_y)
        # (eq:open2) a_C hb(q) < R < a_Y hb(q)
        self.assertLess(self.A_C * hb(self.Q), self.rate)
        self.assertLess(self.rate, a_y * hb(self.Q))
        # (eq:open3) H_234(a_C) < Rc < min(H_134(a_C), H_134(a_Y))
        self.assertAlmostEqual(h(self.A_C, self.A_C), 1.0753403335, places=9)
        self.assertLess(h(self.A_C, self.A_C), self.RC)
        self.assertLess(self.RC, min(h(self.A_C, 0.0), h(a_y, 0.0)))

    def test_task_endpoint_is_a_strictly_improving_witness(self) -> None:
        assert self.k3 is not None and self.task_point is not None
        # Three-mode optimum sits at the identity-absent point (a_C, a_C).
        weights = self.k3.metric.weights
        self.assertAlmostEqual(weights[0], 0.0, places=12)
        self.assertAlmostEqual(weights[1], self.A_C, places=12)
        self.assertAlmostEqual(self.k3.metric.distortion, 0.657316, places=12)
        # All-positive task witness with the certified distortion.
        b_c = self.task_point.weights[1]
        self.assertTrue(all(w > 0.0 for w in self.task_point.weights))
        self.assertAlmostEqual(b_c, 0.3780757053979751, places=14)
        self.assertAlmostEqual(
            self.task_point.distortion, 0.6310809673397869, places=14
        )
        self.assertLess(self.task_point.distortion, self.k3.metric.distortion)

    def test_task_endpoint_is_not_the_global_k4_optimum(self) -> None:
        """The payload endpoint is strictly better: the old wording fails."""
        assert self.payload_point is not None and self.k4 is not None
        self.assertTrue(all(w > 0.0 for w in self.payload_point.weights))
        self.assertAlmostEqual(
            self.payload_point.weights[1], 0.3828360067226760, places=14
        )
        self.assertAlmostEqual(
            self.payload_point.distortion, 0.6308474049505756, places=14
        )
        assert self.task_point is not None
        self.assertLess(
            self.payload_point.distortion, self.task_point.distortion
        )
        # The finite-candidate optimizer selects the payload endpoint.
        self.assertEqual(self.k4.name, "K4-a-right")
        self.assertEqual(self.k4.metric.support, 4)
        self.assertAlmostEqual(
            self.k4.metric.distortion, self.payload_point.distortion, places=14
        )

    def test_corrected_corollary_inequalities_hold(self) -> None:
        """D4* <= D(a_C,b_C) < D3* and D3*-D4* >= 4q(1-q)(a_C-b_C) > 0."""
        assert self.k3 is not None and self.task_point is not None
        assert self.k4 is not None
        d3 = self.k3.metric.distortion
        d4 = self.k4.metric.distortion
        d_witness = self.task_point.distortion
        self.assertLessEqual(d4, d_witness)
        self.assertLess(d_witness, d3)
        b_c = self.task_point.weights[1]
        lower = 4.0 * self.Q * (1.0 - self.Q) * (self.A_C - b_c)
        self.assertGreater(lower, 0.0)
        self.assertGreaterEqual(d3 - d4, lower - 1e-12)


class PositiveCorrelationModelMWitness(unittest.TestCase):
    """q=0.1, y=0.022, eps=0.1, a_C=0.02, a_R=0.021, Rc=0.26."""

    Q, Y, EPS = 0.1, 0.022, 0.1
    A_C, A_R, RC = 0.02, 0.021, 0.26

    def setUp(self) -> None:
        self.rate, self.task = _budgets(self.Q, self.EPS, self.A_C, self.A_R)
        self.assertAlmostEqual(self.task, 0.6758554166854998696, places=15)
        self.assertAlmostEqual(self.rate, 0.00984890746537490565, places=15)

    def test_strict_open_set_conditions(self) -> None:
        h = lambda a, b: evaluate(  # noqa: E731
            self.Q, self.Y, self.EPS, a, b
        ).common_randomness
        a_y = self.Y / (1.0 - self.Q)
        self.assertLess(0.0, self.A_C)
        self.assertLess(self.A_C, a_y)
        self.assertLess(self.A_C * hb(self.Q), self.rate)
        self.assertLess(self.rate, a_y * hb(self.Q))
        self.assertLess(h(self.A_C, self.A_C), self.RC)
        self.assertLess(self.RC, min(h(self.A_C, 0.0), h(a_y, 0.0)))

    def test_descending_roots_lie_in_positive_correlation_half(self) -> None:
        for a in (self.A_C, self.A_R):
            point = optimize_b_at_fixed_a(
                self.Q, self.Y, self.EPS, a, self.RC
            )
            self.assertIsNotNone(point)
            assert point is not None
            b = point.weights[1]
            # H(a, a/2) < Rc < H(a, 0): the descending root is unique.
            h = lambda bb: evaluate(  # noqa: E731
                self.Q, self.Y, self.EPS, a, bb
            ).common_randomness
            self.assertLess(h(a / 2.0), self.RC)
            self.assertLess(self.RC, h(0.0))
            self.assertLess(0.0, b)
            self.assertLess(b, a / 2.0)
            self.assertGreater(a - 2.0 * b, 0.0)  # r > 0
            self.assertTrue(all(w > 0.0 for w in point.weights))

    def test_all_positive_strict_four_mode_optimum_model_m(self) -> None:
        k3 = optimize_k3(
            self.Q, self.Y, self.EPS, self.rate, self.task, self.RC
        )
        k4 = optimize_k4_finite_candidates(
            self.Q, self.Y, self.EPS, self.rate, self.task, self.RC
        )
        self.assertIsNotNone(k3)
        self.assertIsNotNone(k4)
        assert k3 is not None and k4 is not None
        self.assertEqual(k4.metric.support, 4)
        self.assertLess(k4.metric.distortion, k3.metric.distortion)
        a4 = k4.metric.weights[0] + k4.metric.weights[1]
        b4 = k4.metric.weights[1]
        self.assertGreater(a4 - 2.0 * b4, 0.0)  # positive-correlation branch
        self.assertAlmostEqual(k3.metric.distortion, 0.1212, places=12)
        self.assertAlmostEqual(
            k4.metric.distortion, 0.1172209008142478, places=14
        )


if __name__ == "__main__":
    unittest.main()
