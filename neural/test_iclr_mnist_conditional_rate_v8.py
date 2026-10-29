import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("compressai"),
    "PyTorch and CompressAI are required",
)
class ConditionalRateV8Test(unittest.TestCase):
    def test_stateless_offset_schedule_and_ans_roundtrip(self):
        import torch
        from compressai.entropy_models import EntropyBottleneck

        from validation_iclr_mnist.universal_offset_coder import (
            UniversalOffsetEntropyCoder,
            offset_values,
            stateless_level_indexes,
        )

        entropy = EntropyBottleneck(
            2,
            tail_mass=1e-9,
            entropy_coder="ans",
            entropy_coder_precision=16,
        ).eval()
        latent = torch.tensor(
            [
                [
                    [[-1.2, -0.2], [0.3, 1.7]],
                    [[-2.0, -0.6], [0.9, 2.1]],
                ],
                [
                    [[-0.8, 0.1], [0.8, 1.1]],
                    [[-1.3, -0.1], [1.3, 2.8]],
                ],
            ],
            dtype=torch.float32,
        )
        levels = stateless_level_indexes(
            latent.shape, levels=3, seed=20260812, device=latent.device
        )
        receiver_levels = stateless_level_indexes(
            latent.shape, levels=3, seed=20260812, device=latent.device
        )
        torch.testing.assert_close(levels, receiver_levels, atol=0, rtol=0)
        torch.testing.assert_close(
            offset_values(3), torch.tensor([-0.25, 0.0, 0.25]), atol=0, rtol=0
        )

        coder = UniversalOffsetEntropyCoder(
            entropy,
            levels=3,
            symbol_minimum=-16,
            symbol_maximum=16,
        )
        conditioned = coder.encode_decode(latent, levels, conditioned=True)
        marginal = coder.encode_decode(latent, levels, conditioned=False)
        self.assertIsInstance(conditioned.payload, bytes)
        self.assertIsInstance(marginal.payload, bytes)
        torch.testing.assert_close(
            conditioned.symbols, marginal.symbols, atol=0, rtol=0
        )
        torch.testing.assert_close(
            conditioned.decoded_latent, marginal.decoded_latent, atol=0, rtol=0
        )
        self.assertEqual(
            coder.conditional_table_bytes, 3 * coder.marginal_table_bytes
        )

    def test_one_level_is_exact_marginal_identity_control(self):
        import torch
        from compressai.entropy_models import EntropyBottleneck

        from validation_iclr_mnist.universal_offset_coder import (
            UniversalOffsetEntropyCoder,
            stateless_level_indexes,
        )

        entropy = EntropyBottleneck(1, entropy_coder="ans").eval()
        latent = torch.linspace(-2.0, 2.0, steps=16).reshape(1, 1, 4, 4)
        levels = stateless_level_indexes(
            latent.shape, levels=1, seed=7, device=latent.device
        )
        coder = UniversalOffsetEntropyCoder(
            entropy,
            levels=1,
            symbol_minimum=-16,
            symbol_maximum=16,
        )
        conditioned = coder.encode_decode(latent, levels, conditioned=True)
        marginal = coder.encode_decode(latent, levels, conditioned=False)
        self.assertEqual(conditioned.payload, marginal.payload)
        self.assertEqual(
            conditioned.quantized_cdf_bits, marginal.quantized_cdf_bits
        )
        self.assertEqual(conditioned.floating_model_bits, marginal.floating_model_bits)

    def test_escape_tail_matches_compressai_bypass_roundtrip(self):
        import torch
        from compressai.entropy_models import EntropyBottleneck

        from validation_iclr_mnist.universal_offset_coder import (
            UniversalOffsetEntropyCoder,
            stateless_level_indexes,
        )

        entropy = EntropyBottleneck(1, tail_mass=1e-9, entropy_coder="ans").eval()
        latent = torch.tensor([[[[-20.2, -1.0], [1.0, 22.7]]]], dtype=torch.float32)
        levels = stateless_level_indexes(
            latent.shape, levels=3, seed=11, device=latent.device
        )
        coder = UniversalOffsetEntropyCoder(
            entropy,
            levels=3,
            symbol_minimum=-4,
            symbol_maximum=4,
            precision=16,
            escape_tail=True,
        )
        conditioned = coder.encode_decode(latent, levels, conditioned=True)
        marginal = coder.encode_decode(latent, levels, conditioned=False)
        self.assertGreater(conditioned.out_of_support_symbols, 0)
        self.assertGreater(marginal.out_of_support_symbols, 0)
        torch.testing.assert_close(
            conditioned.symbols, marginal.symbols, atol=0, rtol=0
        )
        torch.testing.assert_close(
            conditioned.decoded_latent, marginal.decoded_latent, atol=0, rtol=0
        )
        self.assertLessEqual(coder.max_normalization_residual, 1e-6)

    def test_escape_tail_requires_compressai_rans_precision(self):
        from compressai.entropy_models import EntropyBottleneck

        from validation_iclr_mnist.universal_offset_coder import (
            UniversalOffsetEntropyCoder,
        )

        entropy = EntropyBottleneck(1, entropy_coder="ans").eval()
        with self.assertRaisesRegex(ValueError, "16-bit CDF precision"):
            UniversalOffsetEntropyCoder(
                entropy,
                levels=3,
                precision=15,
                escape_tail=True,
            )


if __name__ == "__main__":
    unittest.main()
