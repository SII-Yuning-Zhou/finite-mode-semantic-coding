"""CPU-only tests for the released v7 frozen-codec configuration."""
from pathlib import Path
import unittest

from run_iclr_mnist_grid import expand_grid
from validation_iclr_mnist.config import _construct, protocol_fingerprint

ROOT = Path(__file__).resolve().parents[1]


class ReleaseConfigurationTest(unittest.TestCase):
    def test_frozen_rate_grid(self):
        entries = expand_grid(ROOT / "configs/iclr_mnist/grid_s1_rate_calibration_v7.json")
        self.assertEqual(len(entries), 4)
        self.assertEqual({x["train"]["rate_weight"] for x in entries}, {0, 200, 5000, 50000})
        for item in entries:
            config = _construct(item)
            self.assertFalse(config.train.mixed_precision)
            self.assertFalse(config.train.allow_tf32)
            self.assertEqual(config.model.decoder_output_activation, "identity")
            self.assertEqual(config.model.decoder_output_postprocess, "clamp_0_1")
            self.assertEqual(config.model.gdn_implementation, "compressai")
            self.assertEqual(config.seed, 20260811)
            self.assertEqual(config.data.split_seed, 20260801)
            self.assertEqual(config.train.steps, 3000)
            self.assertEqual(config.train.batch_size, 512)
            self.assertTrue(protocol_fingerprint(config))


if __name__ == "__main__":
    unittest.main()
