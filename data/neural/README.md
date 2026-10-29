# Frozen MNIST conditional-rate evidence

These files support Fig. 6 and Table IV of the manuscript. They contain all
100 paired observations: four frozen codec rate weights, five offset levels,
and five independently regenerated schedules. The 20 L=1 observations are the
identity controls; the other 80 observations all remain in the release.

- `conditional_rate_v81_raw.csv`: 100 schedule-level observations.
- `conditional_rate_v81_summary.csv`: 20 checkpoint/level groups, each over five schedules.
- `conditional_rate_v81_audit.json`: numerical audit of these exported CSVs.
- `v81_reports/`: all 20 formal per-codec/repetition JSON reports and their
  20 original expanded configurations, plus aggregate/completion/provenance/status.
- `sanitization.json`: source archive/member checksums, excluded log filenames,
  and paths of fields whose local absolute path was made repository-relative.
- `original_server_unit_tests.json`: safe structured summary extracted from
  the archive's unit-test log; **historical server tests, not packaging-host execution**.
- `edge/table_v_transcribed.csv`: **transcription of the four Table V summary
  rows, not original batch-one profile measurements**. Individual profile
  records are not reconstructed or invented from these ranges.
- `frozen_artifacts_manifest.json`: source archive identity, checkpoint byte
  identities and JSON path transformations for the 26 RC2 input files.
- `../../results/iclr_mnist/server_controller_edge_metrics_v1/`: the eight
  original profile reports, expanded configurations and full archived ledger.
  `reproduce.tables` rebuilds Table V from those profiles, then checks the
  retained transcription as an independent formatting/rounding comparison.

## Units and interpretation

`rate_weight` is the codec objective parameter lambda_R; it is not bpp.
`levels` is L. Bit rates and gains use bits per pixel; multiply a gain by
784 to obtain saved bits per 28x28 image. Table sizes are byte counts in the
raw CSV (divide by 1024 for KiB). MSE is a pixel mean on normalized images;
accuracy is a fraction. Repetitions index receiver-known offset schedules at
fixed images and fixed trained weights, not independent network retraining.
The summary uses two-sided Student-t intervals with df=4 and critical value
2.7764451051977987. Per-coefficient offset entropy is ideal i.i.d. accounting;
it is not the stored PRNG seed cost.

The `conditional_codec_seconds` and `marginal_codec_seconds` fields are elapsed
times for the 5000-image coding audit. **They are not the batch-one sender/
receiver means or P95 values in Table V**.

The reports retain the measured zero-gain identity controls, original tail
masses and normalization residuals, all repetitions, and the original
`git_diff_clean=false` provenance. Statistical values and fingerprints were
not modified. Only absolute local paths were sanitized; worker logs were not
distributed. Fingerprints identify the original server artifacts, not hashes
of the path-sanitized public JSON copies.

The source `conditional_rate_v81_audit.json` is a historical analysis export;
its `unit_tests` field describes the original server record. Re-running
`neural/analyze_iclr_mnist_conditional_rate_v81.py` does not execute neural unit
tests and now states that explicitly rather than emitting an unconditional
pass. Local validation/skips are recorded separately.

## Frozen inputs and external dataset

MNIST images are not bundled. Obtain MNIST using the included torchvision
loader or the original dataset source, following the dataset's own terms.
The five frozen `.pt` checkpoints and eight original edge-profile JSON records
are included in RC2 at their original `results/iclr_mnist/` relative paths.
See `../../docs/NEURAL_REPRODUCTION.md` for commands. Original fingerprints and
dirty-worktree provenance remain unchanged. Reaggregating archived timings is
distinct from a new endpoint benchmark; local checkpoint/ANS smoke checks are
reported separately in `../../validation/`.
