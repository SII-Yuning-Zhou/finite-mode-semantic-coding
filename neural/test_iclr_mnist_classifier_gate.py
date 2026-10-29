import unittest

from validation_iclr_mnist.classifier_gate import classifier_health_failures


class ICLRMNISTClassifierGateTest(unittest.TestCase):
    def test_clean_healthy_classifier_passes_with_low_degraded_baseline(self):
        self.assertEqual(
            classifier_health_failures(
                {"clean_accuracy": 0.991, "degraded_7x7_accuracy": 0.2252}
            ),
            [],
        )

    def test_low_clean_accuracy_fails(self):
        failures = classifier_health_failures(
            {"clean_accuracy": 0.97, "degraded_7x7_accuracy": 0.90}
        )
        self.assertTrue(any("clean_accuracy" in item for item in failures))

    def test_malformed_degraded_baseline_fails(self):
        failures = classifier_health_failures(
            {"clean_accuracy": 0.99, "degraded_7x7_accuracy": float("nan")}
        )
        self.assertTrue(any("degraded_7x7_accuracy" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
