# Environments

## Lightweight reproduction

```bash
python -m pip install -r requirements.txt
python run_reproduction.py
```

This path verifies theory and saved measurements, rebuilds tables and plots,
and uses package-relative inputs. It needs Python 3.11 or newer, NumPy,
Matplotlib, mpmath and Pillow.

The recorded Windows validation environment was Python 3.11.4, NumPy 2.4.4,
Matplotlib 3.10.9, mpmath 1.3.0 and Pillow 12.2.0. Optional symbolic/test tools
were SymPy 1.14.0 and pytest 9.0.3. Exact execution records are in
[validation/](../validation/).

## Neural evaluation and profiling

Install compatible PyTorch and torchvision builds for your machine before
installing `neural/requirements.txt`.

| Component | Recorded experiment environment |
|---|---|
| Runtime image | NVIDIA PyTorch 25.02 |
| PyTorch | 2.7.0a0+ecf3bae40a.nv25.02 |
| torchvision | 0.22.0a0 |
| CompressAI | 1.2.8 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 4090 |
| Precision | FP32; AMP and TF32 disabled |

Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing PyTorch when
deterministic CUDA execution is required. CompressAI must provide its ANS
extension.

Endpoint profiling uses batch one, one CPU intra-op/inter-op thread, 20 warm-up
calls and 100 timed calls. Recorded timings describe the saved platform;
running the profiler produces measurements for the reader's own platform.

## CPU model and entropy-coder checks

A compatible Linux environment was used for the published checkpoint smoke
and regression checks:

| Component | Recorded verification environment |
|---|---|
| Python | 3.12.15 |
| PyTorch | 2.6.0+cu126 |
| torchvision | 0.21.0+cu126 |
| CompressAI | 1.2.8 |
| NumPy | 1.26.4 |
| Execution | CPU, one thread |

The five checkpoints passed safe loading, finite-value checks and model
compatibility checks. Small-input neural/coding checks and all four ANS
regression tests passed. The detailed scopes are in
[checkpoint_validation.json](../validation/checkpoint_validation.json) and
[ans_regression_validation.json](../validation/ans_regression_validation.json).
The lightweight Windows run records its optional-dependency skips separately.

## Plotting fonts

Publication PDFs retain their embedded fonts. Portable numeric replots use
Matplotlib's available DejaVu/STIX mathematical fonts; both the exact
publication assets and the underlying numeric data are supplied.

See [MNIST codec](NEURAL_REPRODUCTION.md) for evaluation and profiling commands.
