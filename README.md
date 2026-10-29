# Finite-Mode Semantic Coding for Edge IoT

Code and reproducibility data for **Finite-Mode Semantic Coding for Edge IoT
under Target Distribution and Shared-Randomness Constraints**.

Yuning Zhou, Yunkai Wang, Zhilan Lu, Ziwei Li, Feng Bao, Chao Shen,
Junwen Zhang, and Nan Chi.

This package connects the finite-mode theory to executable numerical checks,
archived paired conditional-rANS measurements, and figure/table reproduction.
This is **version `1.0.0`**, corresponding to the author manuscript snapshot V53.
The manuscript PDF/TeX and its development history are not included in this
software/data package.

- Repository: [SII-Yuning-Zhou/finite-mode-semantic-coding](https://github.com/SII-Yuning-Zhou/finite-mode-semantic-coding)
- Fixed release: [v1.0.0](https://github.com/SII-Yuning-Zhou/finite-mode-semantic-coding/releases/tag/v1.0.0)

No Zenodo DOI has been assigned. The GitHub release identifies this version.

## Quick start: lightweight CPU path

Use Python 3.11 or newer. From an unpacked package root:

```bash
python -m venv .venv
# Activate .venv using your operating system's normal command.
python -m pip install -r requirements.txt
python run_reproduction.py
```

The unified command runs the theoretical witness checks, verifies the archived
MNIST records and regenerates portable scientific plots. It does not download
MNIST, train a neural network, or start a GPU benchmark. New outputs go to
`outputs/`; the supplied evidence under `data/` is read-only input.

Individual verification commands:

```bash
python -m theory.verify
python neural/quick_verify.py
python -m reproduce.check_data --output outputs/figure_source_checks.json
python -m unittest discover -s tests/theory -v
```

See `validation/validation_report.json` for what was actually executed during
packaging, including explicit skips and the tested environment.

## Contents

| Location | Purpose |
|---|---|
| `theory/` | Bernoulli optimizers, finite-candidate and strict-gain certificates, finite-alphabet witness, sampler/simulation programs |
| `data/theory/` | Complete archived point-level synthetic-source tables and certificate records |
| `neural/` | Conditional-rANS, codec training/evaluation and endpoint profiling code; lightweight archived-record checker |
| `configs/` | Frozen protocol and training configurations, including reproduction seeds |
| `data/neural/` | All paired schedule-level observations, reports, artifact provenance and the retained Table V transcription |
| `results/iclr_mnist/` | Four original codec checkpoints, the frozen classifier, eight original endpoint profiles and supporting metadata |
| `data/figures/`, `reproduce/` | Frozen figure scene/source arrays and standalone portable plotting utilities |
| `figures/` | Archived publication figure assets and portable rendering support |
| `docs/` | Code-to-paper mappings, environments, provenance, data meanings and publication checklist |
| `tests/` | Fast theoretical regression checks |

The per-component inventories identify original source files and the adaptations
made for package-relative paths. Original observations and controls are not
filtered by sign, gain or quality.

## What the data represent

- The Bernoulli 90-point archive contains means and confidence intervals from
  the recorded simulation protocol. It is not a saved stream of every source
  symbol or every Monte Carlo replication. The generator and seeds are supplied.
- The MNIST archive contains 100 **schedule-level paired measurements**, including
  20 `L=1` controls and 80 nontrivial comparisons. It is not a redistribution of
  MNIST images or a dump of all training/bitstream events.
- Frozen figure scenes retain the numerical arrays, masks, sampled curves and
  CI geometry needed to replot the displayed results. The independent reader
  is a portable scientific replot; it is not a claim of pixel identity with
  the archived publication artwork.
- Table V is independently rebuilt from **eight original endpoint profile
  reports, 16,000 recorded timed calls and their 80-row ledger**. Its four summary rows are checked against
  the separately retained manuscript transcription. Aggregating these recorded
  measurements does not perform new latency measurements.

## Frozen checkpoints and neural replay

The four original codec checkpoints, frozen classifier checkpoint, eight
endpoint profile reports and their configurations are bundled at the original
repository-relative paths. The checkpoint bytes are preserved; public JSON
copies change only private absolute paths and internal identifiers, with transformations recorded in
`data/neural/frozen_artifacts_manifest.json`. Source fingerprints are retained.

```bash
python neural/verify_frozen_artifacts.py
```

Like-for-like neural replay additionally requires MNIST and an appropriate
PyTorch/torchvision/CompressAI environment. Exact commands and the distinction
between archived evidence, CPU checkpoint smoke tests and full dataset replay
are in `docs/NEURAL_REPRODUCTION.md` and `validation/`.

The default path checks the bundled artifacts and rebuilds the recorded tables
without importing PyTorch. Dataset evaluation, training and new profiling are
opt-in; a fresh training run is a new experiment rather than the frozen run.

## Documentation

- [Theory](docs/THEORY_REPRODUCTION.md)
- [MNIST and endpoint profiling](docs/NEURAL_REPRODUCTION.md)
- [Figures and tables](docs/FIGURE_REPRODUCTION.md)
- [Environments](docs/ENVIRONMENT.md)
- [Publication steps](docs/PUBLISHING_CHECKLIST_ZH.md)
- [Obtaining a version DOI (Chinese)](docs/DOI_ZH.md)
- Chinese entry point: [START_HERE_ZH.md](START_HERE_ZH.md)

## License and citation

Original code and usage documentation: **MIT**, see `LICENSE`.
Author-generated numerical data, trained weights and graphics: **CC BY 4.0**,
see `LICENSE-DATA.md`.
Third-party resources retain their own terms, see `THIRD_PARTY_NOTICES.md`.

Use `CITATION.cff` for author and version metadata, and cite the fixed `v1.0.0`
release above. A SHA-256 file manifest and release-archive checksum accompany
the package. A DOI can be added after a separate Zenodo archival deposit.
