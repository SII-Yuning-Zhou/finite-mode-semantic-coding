# Archive compatibility

The [v1.0.0 release](https://github.com/SII-Yuning-Zhou/finite-mode-semantic-coding/releases/tag/v1.0.0)
and [Zenodo record](https://doi.org/10.5281/zenodo.23774113) are fixed archives.
The current repository adds the public `mnist.py` commands described in
[MNIST codec](NEURAL_REPRODUCTION.md). These commands use the same underlying
codec and coding implementation.

## Stable artifact identities

The original checkpoint bytes, observed values, seeds, configuration and
protocol fingerprints remain unchanged. Original file locations are recorded
in:

- [Frozen artifact manifest](../data/neural/frozen_artifacts_manifest.json)
- [Neural inventory](neural_inventory.json)
- [Figure inventory](figure_inventory.json)
- [Source provenance](../SOURCE_PROVENANCE.json)

These records retain historical experiment labels so that an archived report
can still be matched to its checkpoint and expanded configuration. Those
labels are identifiers, not publication-venue claims. Dependency and license
notices are maintained in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Coding settings

The saved training configuration contains an earlier carrier-evaluation
support of [-64, 64]. The paired frozen-offset experiment uses [-128, 128],
16-bit CDFs and explicit escape tails under
`UQ-CONDITIONAL-RANS-ESCAPE-V2`. The public `evaluate` command selects these
frozen-offset settings explicitly.

The supplied frozen classifier is identified by the artifact manifest. Its
preparation configuration name and output-model version are separate labels;
the original records preserve both. Each codec checkpoint is a complete
research container, including codec, classifier, discriminator and
configuration state. Its file size is therefore distinct from the endpoint
encoder/entropy/CDF storage in Table V.

For byte-identical checkpoint identity, the original binary metadata also
retain the historical field
`classifier_pretraining[0].loaded_frozen_classifier`. Replay loads the supplied
weight tensors; it does not require that old location to exist. The validation
report records only this field's name, not its private path value.

<details>
<summary>Original names and commands in the fixed v1.0.0 archive</summary>

The archive's historical namespace is `iclr_mnist`. Reader-facing commands in
the current checkout use `mnist.py`; the archive retains these original paths:

```text
neural/validation_iclr_mnist/
configs/iclr_mnist/grid_s1_rate_calibration_v7.json
configs/iclr_mnist/classifier_reference_v1.json
results/iclr_mnist/
```

The four codec checkpoints have the pattern:

```text
results/iclr_mnist/s1_rate_calibration_v7/s1_rate_calibration_v7_s20260811_lr{rate}_continuous/final.pt
```

Here `{rate}` is one of `0`, `200`, `5000` and `50000`. The classifier and
endpoint profile locations are:

```text
results/iclr_mnist/classifiers/iclr_gray_resnet18_small_v2.pt
results/iclr_mnist/server_controller_edge_metrics_v1/
```

The fixed archive supports the lightweight command:

```bash
python run_reproduction.py
```

For a paired audit using the fixed archive's original entrypoint:

```bash
python neural/audit_iclr_mnist_conditional_rate_v8.py --grid configs/iclr_mnist/grid_s1_rate_calibration_v7.json --rate-weight 5000 --levels 1 3 7 15 31 --maximum-samples 5000 --batch-size 512 --symbol-minimum -128 --symbol-maximum 128 --precision 16 --escape-tail --schedule-repetition 0 --device cuda:0 --output outputs/mnist/rate_5000_rep_0.json
```

For endpoint profiling using the fixed archive:

```bash
python neural/profile_iclr_mnist_edge_deployment.py --rate-weight 5000 --device cpu --threads 1 --levels 1 3 7 15 31 --samples 128 --warmup 20 --repetitions 100 --output-dir outputs/edge_profile
```

The wrapper and original entrypoint select the same coding protocol. New
training through the public CLI derives a separate output configuration and
does not replace these archived inputs.

</details>

## Historical validation records

The validation directory retains the original preparation and execution
reports, including their recorded environments, pass/skip outcomes and
original dirty-worktree provenance. New public naming does not relabel those
historical executions. The current source manifest identifies the current
checkout; the release tag and Zenodo archive identify the fixed version.
