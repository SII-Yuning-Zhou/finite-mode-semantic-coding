# Figures and tables

The repository provides exact publication PDFs and portable numeric replots.
Run from the repository root:

```bash
python -m reproduce.check_data --output outputs/data_checks.json
python -m reproduce.figures --output outputs/figures
python -m reproduce.tables --output outputs/tables
```

The figure command writes PDF, SVG and PNG for Figs. 2–6, plus
`replot_checks.json`. Use `--figures 3 6` to render a subset. Outputs go to
the requested directory; supplied source data remain inputs.

## Publication assets and numeric replots

- [figures/publication/](../figures/publication/) contains the exact six figure
  PDFs used in the manuscript, with PNG viewing previews.
- `reproduce/figures.py` reads frozen source arrays and preserves curves,
  masks, observation centers, confidence intervals, units, ranges, markers
  and legend meanings. It uses open DejaVu/STIX font fallbacks.
- Fig. 1 is a fixed system schematic and is supplied as an archived asset.
  The numeric plotting command renders Figs. 2–6.

Numeric replotting needs NumPy and Matplotlib. It uses the original arrays,
rather than values digitized from the paper. The publication PDFs preserve the
journal typesetting; the portable reader preserves the scientific contents.

## Figure and table mapping

| Item | Source data or record | Entry point |
|---|---|---|
| Fig. 1 | `figures/publication/fig1.pdf` | fixed system diagram |
| Fig. 2 | `data/figures/fig2_scene.json`; geometry at `(q,y,epsilon,R,C,Rc)=(.18,.10,.235,.11,.90,.78)` | `reproduce.figures --figures 2` |
| Fig. 3 | `fig3_scene.json`; `data/figures/source/k4_rrc_phase_diagram_v2.csv` | `reproduce.figures --figures 3` |
| Fig. 4 | `fig4_scene.json`; Bernoulli simulation, resource-curve and phase CSVs | `reproduce.figures --figures 4` |
| Fig. 5 | `fig5_scene.json`; all 90 Bernoulli simulation rows | `reproduce.figures --figures 5` |
| Fig. 6 | `fig6_scene.json`; paired MNIST raw/summary CSVs | `reproduce.figures --figures 6` |
| Table I | finite-alphabet certificate and vertices in `data/theory/general_finite_alphabet/` | [Theory reproduction](THEORY_REPRODUCTION.md) |
| Table II | analytic face formulas | theory solver and manuscript |
| Table III | model and evaluation settings | [MNIST codec](NEURAL_REPRODUCTION.md) |
| Table IV | all five L=31 schedule repetitions at each rate weight | `reproduce.tables` |
| Table V | eight original endpoint profiles and retained table transcription | `reproduce.tables` |

The [figure inventory](figure_inventory.json) gives the exact source-file
mapping. The plotting check retains 90 observed means and 90 CIs in Fig. 4,
360 observed means in Fig. 5 (270 resource CIs plus sampler means), and
80 means and 80 CIs in Fig. 6. It also checks all 6,309 positive phase-map values
against the original 20,402-row CSV.

## Table reproduction

Table IV is rebuilt from the five paired schedule repetitions at each of the
four frozen operating points. Table V is rebuilt from the eight original
endpoint profiles and their 80-row ledger, then compared with the separately
labelled `data/neural/edge/table_v_transcribed.csv` at its displayed precision.

Timing statistics are reconstructed from the saved profile records.
[EDGE_RECORDS.md](EDGE_RECORDS.md) describes their fields and units.
`conditional_codec_seconds` in the whole-dataset coding audit and batch-one
endpoint encoding latency are separate measurements.

## Source-array format

Scene files use `finite-mode-frozen-scientific-scene-v1`: canvas dimensions,
axes limits/scales/ticks/labels, line arrays, mesh coordinates, masks, contour,
fill and CI paths, point offsets, top-axis mappings and legends. A `null`
coordinate represents an original masked/non-finite location; the reader
restores the mask or line break rather than replacing it with zero.

The source CSVs and scene arrays retain all controls and repetitions.
The detailed schema and original mappings remain in the inventories; simulation,
dataset evaluation and endpoint profiling can also regenerate observations
through the theory and [MNIST](NEURAL_REPRODUCTION.md) programs.
