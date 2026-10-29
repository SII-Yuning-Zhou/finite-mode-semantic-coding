# MNIST conditional rANS and endpoint reproduction

## Included, runnable now

Run from the release root with Python 3.11 or newer:

```bash
python neural/quick_verify.py
python neural/verify_frozen_artifacts.py
python -m reproduce.tables --output outputs/tables
```

This standard-library command checks all 100 paired observations, the 20
L=1 controls, the five schedules per group, exact byte-table scaling, the
Student-t intervals, the formal report protocol, and the four Table IV rows.
It does not retrain a neural network or re-time an endpoint. No observations
are selected by their sign or quality.

The original archive protocol is `UQ-CONDITIONAL-RANS-ESCAPE-V2`. Current neural
code is in `neural/validation_iclr_mnist/`; the helper modules in
`neural/validation_semcom/` provide schedule and metric definitions. This is the
formal rANS/escape protocol, not the earlier Huffman/AMP pilot workflow.
Legacy compatibility classes remain in the source, but the released v7 configs
select CompressAI GDN/IGDN, linear synthesis, decode-only clamp and FP32 codec
training. The scripts have no dependency on Codex or private skill folders.

## Optional neural execution environment

The measured server environment was NVIDIA PyTorch image 25.02:

- `torch=2.7.0a0+ecf3bae40a.nv25.02`, `torchvision=0.22.0a0`, CUDA 12.8;
- `compressai=1.2.8`; NVIDIA GeForce RTX 4090;
- source/target split: 55000/5000 training/validation, PCG64 seed 20260801;
- codec seed 20260811; four rate weights 0, 200, 5000 and 50000;
- codec width 48; latent 12x4x4; 3000 steps; batch 512;
- codec AMP and TF32 disabled. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before
  importing PyTorch when deterministic CUDA behavior is required.

Install a compatible PyTorch/torchvision build for your machine, then install
`neural/requirements.txt`. The archived NVIDIA alpha image is not claimed to be
identical to an arbitrary stable pip installation. CompressAI must expose its
ANS extension. This package does not automatically install or replace a GPU
runtime.

```bash
python -m unittest discover -s neural -p "test_release_config.py" -v
python -m unittest discover -s neural -p "test_iclr_mnist_classifier_gate.py" -v
python -m unittest discover -s neural -p "test_iclr_mnist_conditional_rate_v8.py" -v
```

The ANS tests deliberately skip if PyTorch/CompressAI are unavailable. A skip
is not successful bitstream validation. The Windows lightweight environment
does not have the optional neural dependencies. The recorded RC2 checkpoint validation uses
a separately recorded, existing Linux CPU environment; consult
`validation/checkpoint_validation.json` for its actual checks and environment.
The four original conditional-rANS regression tests were also run in that
Linux CPU environment: four passed, zero skipped; see
`validation/ans_regression_validation.json`. A Windows-only run without the
optional packages still reports its own four skips rather than borrowing
the Linux result.

## Bundled original frozen checkpoints

The following actual artifact paths are referenced by the archived reports:

```text
results/iclr_mnist/s1_rate_calibration_v7/s1_rate_calibration_v7_s20260811_lr0_continuous/final.pt
results/iclr_mnist/s1_rate_calibration_v7/s1_rate_calibration_v7_s20260811_lr200_continuous/final.pt
results/iclr_mnist/s1_rate_calibration_v7/s1_rate_calibration_v7_s20260811_lr5000_continuous/final.pt
results/iclr_mnist/s1_rate_calibration_v7/s1_rate_calibration_v7_s20260811_lr50000_continuous/final.pt
results/iclr_mnist/classifiers/iclr_gray_resnet18_small_v2.pt
```

All five files are bundled at these relative paths in version 1.0.0. Their original bytes
are retained and linked to the formal v8.1 reports and endpoint profiles by
checkpoint fingerprints. The input archive and public-file mappings are in
`data/neural/frozen_artifacts_manifest.json`. Download this version from
https://github.com/SII-Yuning-Zhou/finite-mode-semantic-coding/releases/tag/v1.0.0;
the source repository is https://github.com/SII-Yuning-Zhou/finite-mode-semantic-coding.

Keep these relative paths or explicitly use the relevant CLI checkpoint
argument. Checkpoint structure/finite-tensor checks use CPU
`torch.load(..., weights_only=True)` without an unsafe pickle fallback.
The optional checkpoint smoke-test command is:

```bash
python neural/check_frozen_checkpoints.py --run-cpu-smoke --output outputs/checkpoint_validation.json
```

It is separate from the lightweight default and from evaluation on all 5,000
validation images. Output goes to `outputs/`, preserving the bundled reports.

Each codec checkpoint is a complete research container, also holding the
frozen classifier, discriminator and configuration. Its file size is distinct
from the endpoint encoder/entropy/CDF storage measured in Table V.
The original binary metadata retain one historical classifier-location field,
`classifier_pretraining[0].loaded_frozen_classifier`. It is kept to preserve
checkpoint byte identities and is not needed as a local path for replay. The
checkpoint validation report records the field name without copying its value.

The preparation config named `classifier_reference_v1.json` is the original
preparation input; the frozen output used by the formal v7/v8.1 experiment is
the **v2** classifier path above. Its name is not permission to substitute an
earlier failed/degraded-input classifier. The supplied preparation code trains
and validates the clean-domain classifier health criterion.

## Evaluate the frozen weights

The release root is the working directory, not the `neural/` directory:

```bash
python neural/audit_iclr_mnist_conditional_rate_v8.py --grid configs/iclr_mnist/grid_s1_rate_calibration_v7.json --rate-weight 5000 --levels 1 3 7 15 31 --maximum-samples 5000 --batch-size 512 --symbol-minimum -128 --symbol-maximum 128 --precision 16 --escape-tail --schedule-repetition 0 --device cuda:0 --output runs/neural/rate_5000_rep_0.json
```

Repeat for all four weights and repetitions 0..4 to regenerate the formal
100 comparisons. The CLI filename retains `v8` for compatibility; the
`--escape-tail` flag selects the v8.1 `ESCAPE-V2` protocol. MNIST is downloaded
through torchvision under `data_external/mnist`; the official test set is not
used by this validation experiment.

The v7 grid's `evaluation.symbol_minimum/maximum` fields retain the training
carrier's earlier [-64,64] evaluation support. The formal frozen-offset v8.1
audit explicitly overrides them with [-128,128], 16-bit precision and escape
tails through the CLI above. The original grid/config fingerprints are not
rewritten to conceal that distinction.

To generate new training runs (distinct from evaluating the supplied weights):

```bash
python neural/prepare_iclr_mnist_classifier.py --config configs/iclr_mnist/classifier_reference_v1.json --output results/iclr_mnist/classifiers/iclr_gray_resnet18_small_v2.pt --device cuda:0
python neural/run_iclr_mnist_grid.py --grid configs/iclr_mnist/grid_s1_rate_calibration_v7.json --device cuda:0 --action train-evaluate
```

These are explicitly expensive opt-in commands. The quick reproduction path
does not run them.

## Endpoint storage/latency (Table V)

The profiling implementation, aggregator and all eight original profile JSON
files (4 codecs x CPU/CUDA) are included. Their directory is:

```text
results/iclr_mnist/server_controller_edge_metrics_v1
```

Each profile preserves its timing sequences, memory accounting, hardware,
measurement protocol and checkpoint identity. Its public JSON copy changes
only private absolute paths and internal identifiers. The 80-row original ledger is included as well.
Field definitions and aggregation units are in [EDGE_RECORDS.md](EDGE_RECORDS.md).

Reaggregate the existing measurements without importing PyTorch or running a
new benchmark:

```bash
python neural/aggregate_edge_deployment_metrics.py --root results/iclr_mnist/server_controller_edge_metrics_v1 --output outputs/edge/edge_deployment_metrics.json
python -m reproduce.tables --output outputs/tables
```

The second command builds all four Table V summary rows from the original
profiles and compares them with `data/neural/edge/table_v_transcribed.csv`.
The transcription remains explicitly labelled; it is not substituted for the
original reports. Reaggregation is not a new latency measurement.

To collect new measurements on a reader's platform, with optional neural
dependencies installed:

```bash
python neural/profile_iclr_mnist_edge_deployment.py --rate-weight 5000 --device cpu --threads 1 --levels 1 3 7 15 31 --samples 128 --warmup 20 --repetitions 100 --output-dir runs/edge_profile
python neural/aggregate_edge_deployment_metrics.py --root runs/edge_profile --output runs/edge_profile/edge_deployment_metrics.json
```

Run all four rates and CPU/CUDA paths before comparing with all Table V ranges.
Batch one, FP32, one intra-op/inter-op CPU thread, setup excluded, 20 warm-up
and 100 timed calls per path are the reported protocol. Timing is
platform-dependent; new measurements are not the archived 2026 measurements.

## Optional author-side artifact export

No remote connection is performed by this package. The author may execute this
script on the original server, giving an archive path outside the source repo:

```bash
bash neural/export_missing_artifacts.sh /absolute/path/to/original/research_repo /absolute/path/outside/research_repo/frozen_mnist_artifacts.tar.gz --list-only
# Remove --list-only only after reviewing the printed file list.
```

The script enumerates the five explicit checkpoints, classifier/calibration
metadata and the eight expected profile records plus ledgers; it never removes
or overwrites source files and refuses an archive inside the repository.
Any newly exported transfer bundle needs metadata/path inspection before it is
added to a public release. The bundled original inputs have the inspection and
transformation records described above.

## Data and dependency scope

All generated JSON/CSV observations and numerical seeds are preserved. The
public JSON copies remove only absolute development path prefixes and exclude
22 logs; scientific values and the original `git_diff_clean=false` flag remain.
Original source archive/member checksums are in `sanitization.json`. Fingerprints
refer to original server files and are not relabelled as hashes of sanitized
copies.

PyTorch, torchvision, CompressAI, NumPy, Matplotlib and psutil are imported
dependencies, not redistributed source here; follow their respective licenses.
MNIST images and third-party pretrained weights are not bundled. The five
included checkpoints were trained by the authors. This version is distributed
through the public GitHub repository and fixed release linked above. Running
these reproduction scripts does not publish files or assign a DOI.
