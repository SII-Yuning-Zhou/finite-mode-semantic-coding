# MNIST codec

Evaluate the supplied codecs, compare conditional and marginal rANS, and
measure endpoint storage and latency through the public `mnist.py` interface.
Run the commands below from a current clone of this repository.

## Quick checks

```bash
python mnist.py list-models
python mnist.py verify
python neural/quick_verify.py
python -m reproduce.tables --output outputs/tables
```

These commands check the bundled checkpoints and reports, verify all 100
paired schedule observations, and reconstruct Tables IV–V from the saved
measurements. They run with the lightweight environment in
[ENVIRONMENT.md](ENVIRONMENT.md).

## Installation

For neural evaluation, first install a compatible PyTorch and torchvision
build for your CPU or GPU, then install the neural dependencies:

```bash
python -m pip install -r neural/requirements.txt
```

The codec uses [CompressAI](https://github.com/InterDigitalInc/CompressAI)
GDN/IGDN layers and its entropy-coding implementation. The original
experiments used CompressAI 1.2.8. Full runtime versions are listed in
[ENVIRONMENT.md](ENVIRONMENT.md); dependency licenses are listed in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Reference setup

| Component | Setting |
|---|---|
| Source images | Bilinear 28 × 28 → 7 × 7 → 28 × 28; antialiased downsampling |
| Data split | 55,000 train / 5,000 validation; PCG64 seed 20260801; sorted original indices |
| Source/target batches | Independently shuffled degraded-source and clean-target pools |
| Codec seed | 20260811 |
| Codec | Width 48; latent 12 × 4 × 4; GDN/IGDN; linear synthesis and decode-output clamp |
| Frozen classifier | Grayscale ResNet-18, width 32 |
| Optimization | 3,000 steps; batch 512; codec learning rate 0.005 |
| Precision | FP32; AMP and TF32 disabled |

The supplied [configurations](../configs/) retain all optimizer rates, loss
weights, critic settings and random schedules used for these operating points.

## Pretrained models

| Model | Rate weight | Codec configuration |
|---|---:|---|
| MNIST codec | 0 | Width 48; latent 12 × 4 × 4 |
| MNIST codec | 200 | Width 48; latent 12 × 4 × 4 |
| MNIST codec | 5,000 | Width 48; latent 12 × 4 × 4 |
| MNIST codec | 50,000 | Width 48; latent 12 × 4 × 4 |

All four codecs and the frozen classifier are included in the
[v1.0.0 release](https://github.com/SII-Yuning-Zhou/finite-mode-semantic-coding/releases/tag/v1.0.0)
and [Zenodo archive](https://doi.org/10.5281/zenodo.23774113).
`python mnist.py list-models --json` lists model identifiers, rate weights,
availability and checkpoint sizes. The
[artifact manifest](../data/neural/frozen_artifacts_manifest.json) records the
original file paths and byte fingerprints; `mnist.py verify` checks them.

To check model compatibility and a small CPU round trip:

```bash
python mnist.py checkpoints --run-cpu-smoke --output outputs/checkpoint_validation.json
```

This loads the supplied files with CPU `weights_only=True`, checks finite
tensors and model compatibility, and exercises neural and entropy-coding
paths. The recorded results are in
[checkpoint_validation.json](../validation/checkpoint_validation.json).

## Evaluate a codec

```bash
python mnist.py evaluate --rate-weight 5000 --device cuda:0 --schedule-repetition 0 --output outputs/mnist/rate_5000_rep_0.json
```

Use `--device cpu` for CPU evaluation. The public command selects the
escape-aware conditional-rANS protocol, 16-bit CDFs and symbol support
[-128, 128]. Defaults are 5,000 validation images, batch 512 and offset levels
`1 3 7 15 31`.

Repeat for the four rate weights and schedule repetitions `0 1 2 3 4` to
recreate the full 100-point audit. Each comparison uses the same symbols and
reconstruction for the conditional and marginal coders. The L=1 setting is
the identity control.

MNIST is obtained through torchvision under `data_external/mnist`. The
experiment uses 55,000 training images and 5,000 validation images; the official
test split is not used. Partition and training seeds remain in the supplied
configurations.

## Endpoint profiling

To measure a codec on your own machine:

```bash
python mnist.py profile --rate-weight 5000 --device cpu --threads 1 --levels 1 3 7 15 31 --samples 128 --warmup 20 --repetitions 100 --output-dir outputs/edge_profile
python neural/aggregate_edge_deployment_metrics.py --root outputs/edge_profile --output outputs/edge_profile/edge_deployment_metrics.json
```

Use `--device cuda:0` for the CUDA path. The command records batch-one encode
and decode times, CDF storage, tensor state and memory statistics. Run all four
rate weights on both devices to form a complete ledger. Measurement fields and
units are documented in [EDGE_RECORDS.md](EDGE_RECORDS.md).

The supplied eight profiles contain 16,000 timed calls and an 80-row resource
ledger. To rebuild their Table V summary without running a new benchmark:

```bash
python -m reproduce.tables --output outputs/tables
```

## Train new models

The training command derives a new configuration and writes to a separate
run directory:

```bash
python mnist.py train --device cuda:0 --output-dir runs/mnist_training
```

Use `--rate-weight` for one operating point or `--shard-index` and
`--shard-count` to select a shard of the sweep.

The bundled classifier is the reference for the codec sweep. Classifier
preparation can also be run independently:

```bash
python mnist.py prepare-classifier --device cuda:0 --output runs/classifier/mnist_classifier.pt
```

New outputs remain separate from the supplied frozen weights. See
`python mnist.py COMMAND --help` for each command's options.

## Protocol and archive details

The codec uses FP32, CompressAI GDN/IGDN, linear synthesis and an output clamp
at decoding. The archived coding protocol is
`UQ-CONDITIONAL-RANS-ESCAPE-V2`. The four original conditional-rANS regression
tests passed on CPU in the recorded compatible environment; see
[ans_regression_validation.json](../validation/ans_regression_validation.json).

Original data, checkpoints and experiment fingerprints are retained.
[ARCHIVE_COMPATIBILITY.md](ARCHIVE_COMPATIBILITY.md) explains the fixed-release
entrypoints, historical path identifiers and exact protocol mapping.
