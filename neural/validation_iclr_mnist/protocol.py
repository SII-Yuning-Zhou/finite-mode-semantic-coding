"""Lightweight frozen identifiers for the operational entropy protocol."""

COMPRESSAI_PACKAGE_VERSION = "1.2.8"
COMPRESSAI_ENTROPY_PROTOCOL = (
    "compressai_mode_conditioned_block_ans_cdf16_v2"
)

# The paper reports normalized [0,1] MSE while tabulating raw lambda values up
# to 50,000. Its empirical dD/dR slopes are compatible with applying lambda in
# 8-bit intensity units. Keep fidelity, WGAN, and CE in their written
# normalized units and convert only the tabulated rate weight.
ICLR_DISTORTION_REDUCTION = "pixel_mean"
ICLR_RATE_WEIGHT_SCALE = 1.0 / (255.0 ** 2)
ICLR_LOSS_PROTOCOL = "normalized_mse_raw_lambda_div_255sq_v1"
