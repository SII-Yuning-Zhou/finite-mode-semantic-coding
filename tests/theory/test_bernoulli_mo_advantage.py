"""Unit tests for the synthetic M/O strict-advantage audit (torch-free)."""

from __future__ import annotations

import unittest

import numpy as np

from theory.audit_bernoulli_mo_advantage import (
    WITNESS,
    delta_h_float,
    exact_delta_h,
    scan_ab_region,
    sweep_rc_schedules,
    witness_report,
)
from theory.bernoulli_mode_limited import (
    hb,
    optimize_k3,
    optimize_k4_finite_candidates,
    weights_from_ab,
)


class DeltaHProperties(unittest.TestCase):
    QX, QY, EPS = 0.18, 0.10, 0.235

    def _weights(self, a: float, b: float, qx=None, qy=None) -> np.ndarray:
        weights = weights_from_ab(qx or self.QX, qy or self.QY, a, b)
        self.assertGreaterEqual(float(np.min(weights)), -1e-12)
        return weights

    def test_delta_h_nonnegative_over_feasible_region(self) -> None:
        for a in np.linspace(0.0, 1.0, 11):
            for b in np.linspace(0.0, float(a), 11):
                weights = weights_from_ab(self.QX, self.QY, float(a), float(b))
                if np.min(weights) < -1e-12:
                    continue
                self.assertGreaterEqual(
                    delta_h_float(self.QX, self.EPS, weights), -1e-12
                )

    def test_delta_h_vanishes_for_single_mode_schedules(self) -> None:
        # Identity-only: feasible when qy = qx (w3 = w4 = 0).
        weights = self._weights(1.0, 0.0, 0.18, 0.18)
        self.assertAlmostEqual(
            delta_h_float(0.18, self.EPS, weights), 0.0, places=12
        )
        # Flip-only: feasible when qy = 1 - qx.
        weights = self._weights(1.0, 1.0, 0.18, 0.82)
        self.assertAlmostEqual(
            delta_h_float(0.18, self.EPS, weights), 0.0, places=12
        )

    def test_delta_h_positive_for_mixed_informative_modes(self) -> None:
        # Identity+flip mixture: individually fully informative, but the
        # output-only receiver cannot tell whether Y was flipped.
        qx, qy = 0.1, 0.2
        b = (qy - qx) / (1.0 - 2.0 * qx)
        weights = self._weights(1.0, b, qx, qy)
        self.assertGreater(delta_h_float(qx, self.EPS, weights), 1e-6)

    def test_delta_h_vanishes_for_pure_constant_schedules(self) -> None:
        # a = 0: only constant modes, Y carries no information at all.
        weights = self._weights(0.0, 0.0)
        self.assertAlmostEqual(
            delta_h_float(self.QX, self.EPS, weights), 0.0, places=12
        )

    def test_delta_h_strictly_positive_for_mixed_schedule(self) -> None:
        # Feasible at the manuscript slice: qx*a + (1-2*qx)*b <= qy.
        weights = self._weights(0.3, 0.05)
        self.assertGreater(delta_h_float(self.QX, self.EPS, weights), 1e-3)

    def test_exact_fraction_path_matches_float(self) -> None:
        cases = [
            (self.QX, self.QY, 0.3, 0.05),
            (0.18, 0.18, 1.0, 0.0),  # identity-only
            (self.QX, self.QY, 0.0, 0.0),
            (0.1, 0.3, 0.8, 0.2),
        ]
        for qx, qy, a, b in cases:
            weights = self._weights(a, b, qx, qy)
            exact = exact_delta_h(qx, self.EPS, weights)
            fast = delta_h_float(qx, self.EPS, weights)
            self.assertAlmostEqual(exact, fast, places=12)

    def test_mode_aware_never_exceeds_output_only(self) -> None:
        # H(S|Y,V) <= H(S|Y): conditioning on V cannot hurt on average.
        rows = scan_ab_region(self.QX, self.QY, self.EPS, steps=10)
        for row in rows:
            self.assertLessEqual(
                row["h_mode_aware"], row["h_output_only"] + 1e-12
            )


class WitnessAndSweep(unittest.TestCase):
    def test_witness_matches_pinned_k4_optimum(self) -> None:
        report = witness_report()
        self.assertIn("K3", report)
        self.assertIn("K4", report)
        # Pinned values from test_k4_open_set_counterexample.
        self.assertAlmostEqual(report["K3"]["distortion"], 0.1212, places=12)
        self.assertAlmostEqual(
            report["K4"]["distortion"], 0.1172209008142478, places=14
        )
        self.assertGreater(report["delta_d_k3_minus_k4"], 0.0)
        # The witness schedule mixes informative and constant modes, so
        # receiver-known mode information strictly reduces task uncertainty.
        self.assertGreater(report["K4"]["delta_h"], 1e-6)
        self.assertAlmostEqual(
            report["K4"]["delta_h"],
            report["K4"]["delta_h_exact_fraction"],
            places=12,
        )

    def test_strict_regions_overlap_on_rc_sweep(self) -> None:
        rows = sweep_rc_schedules(
            0.18, 0.10, 0.235, 0.11, 0.90, 0.40, 1.20, 0.10
        )
        k4 = [r for r in rows if r["k"] == 4]
        strict_d = {r["rc"] for r in k4 if r.get("delta_d_k3_minus_k4", 0.0) > 1e-9}
        strict_h = {r["rc"] for r in k4 if r["delta_h"] > 1e-9}
        self.assertTrue(strict_d, "expected a nonempty strict DeltaD region")
        self.assertTrue(strict_h, "expected a nonempty strict DeltaH region")
        self.assertTrue(
            strict_d & strict_h,
            "the strict DeltaD and strict DeltaH regions must overlap",
        )


if __name__ == "__main__":
    unittest.main()
