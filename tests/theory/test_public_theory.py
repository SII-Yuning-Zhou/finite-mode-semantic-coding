"""Fast interval/archived-data checks for the portable public release."""
import unittest
from theory.verify import run_checks


class PortableTheoryCertificates(unittest.TestCase):
    def test_intervals_sampler_and_complete_archive(self):
        result = run_checks()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["archived_simulation_points_checked"], 90)
        self.assertEqual(result["nonbernoulli_vertex_system_counts"]["ambiguous"], 0)


if __name__ == "__main__":
    unittest.main()
