# Environments

## Fast CPU reproduction

The lightweight validation uses Windows and Python 3.11.4 with NumPy 2.4.4,
Matplotlib 3.10.9, mpmath 1.3.0, Pillow 12.2.0, SymPy 1.14.0 and pytest 9.0.3.
The quick path is designed to avoid PyTorch/CUDA/CompressAI and use package-relative
paths. A clean-directory execution report is supplied in `validation/` after testing.

Install the lightweight requirements in a virtual environment. Original paper
neural training and archived records use a different environment; do not conflate
the two when comparing exact neural outputs or runtime measurements.

## Paper neural environment

The reported experiment environment is FP32, PyTorch 2.7.0a0, torchvision
0.22.0a0, CompressAI 1.2.8, CUDA 12.8 and NVIDIA GeForce RTX 4090. Profiling uses
batch one, one CPU intra-op/inter-op thread, 20 warm-up calls and 100 timed calls.
Runtime values are measurements on the recorded platform, not a timing guarantee
for a reader's machine.

See `NEURAL_REPRODUCTION.md` for installation, exact configurations, checkpoint
inputs and optional full evaluation/training. The current Windows packaging
environment has no PyTorch, torchvision or CompressAI; it does not provide a
new neural training, rANS roundtrip or GPU profiling validation.

## Additional RC2 checkpoint validation

An existing Linux environment is used separately for CPU checkpoint loading
and smoke checks: Python 3.12.15, PyTorch 2.6.0+cu126, torchvision 0.21.0+cu126,
CompressAI 1.2.8 and NumPy 1.26.4. No packages are installed into or removed from
that environment by the release builder. CPU execution does not require a GPU.
This runtime differs from the historical NVIDIA 25.02 training image.

The actual checks, outcomes and any skips are recorded in
`validation/checkpoint_validation.json`. A small-input checkpoint smoke test
is separate from replaying the 5,000-image MNIST evaluation and from measuring
new CPU/CUDA timing distributions. The lightweight entry point never installs
this optional runtime, downloads images or starts training.

## Fonts

The original publication PDFs retain their embedded fonts. Portable scientific
replots use the fonts available to Matplotlib, with open mathematical font
fallbacks; numerical data and CI meanings are retained independently of fonts.
Do not infer that a portable replot must be pixel-identical to a publication PDF.
