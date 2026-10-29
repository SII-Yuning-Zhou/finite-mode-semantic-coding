# Endpoint profile records and units

The eight profile reports in
`results/iclr_mnist/server_controller_edge_metrics_v1/` cover four codec rate
weights (0, 200, 5000 and 50000) on the recorded CPU and CUDA paths. Each report
contains five offset levels (1, 3, 7, 15 and 31) and both marginal and conditional
coding paths: 80 ledger rows in total. Each row has separate encode/decode
timing distributions.

## Fields

| Field | Meaning and unit |
|---|---|
| `rate_weight`, `seed` | Training operating-point parameter and fixed codec seed; not bit rates |
| `checkpoint_sha256` | Identity of the original frozen checkpoint bytes |
| `hardware` | Recorded device, platform, logical/physical CPU counts and GPU metadata |
| `measurement_protocol` | Batch/thread/precision, warm-up and sampling settings |
| `model_resources` | Parameter counts, tensor-state byte counts and convolution MACs; model/component scope is explicit in the field name |
| `levels[].levels` | Offset cardinality L |
| `levels[].conditional_table_bytes`, `marginal_table_bytes` | Stored CDF bytes; divide by 1024 for KiB |
| `levels[].paths.*.encode`, `.decode` | Per-image timing statistics for the selected path |
| `raw_seconds` | Original individual timed-call durations, in seconds |
| `mean_ms`, `median_ms`, `p95_ms` | Summary timing statistics, in milliseconds |
| `sender_memory`, `receiver_memory` | Loaded-process RSS, sampled workload increments and CUDA allocation statistics, in bytes |

Each endpoint distribution has 100 timing calls after 20 warm-up calls. The
protocol uses batch one, one intra-op and inter-op CPU thread, FP32, AMP off,
TF32 off, 128 validation images and a 32-bit frame. Setup is excluded. Sender
timing includes neural encoding, shared scheduling, quantization, indexed rANS
and framing; receiver timing includes rANS, neural decoding and output transfer.
See the profiling implementation for synchronization and memory sampling.

The recorded host is x86_64, with 128 logical and 64 physical CPUs; an exact
CPU model name is not present in the source metadata. The GPU is recorded as
NVIDIA GeForce RTX 4090. Hardware metadata are retained as recorded, rather
than inferred from other machines or from the reported timings.

## Reaggregation

```bash
python -m reproduce.tables --output outputs/tables
```

This rebuilds the 80-row ledger, then selects the conditional path at L=1 and
L=31. For each CPU/CUDA and L pair, it takes the minimum/maximum mean and p95
across all four rate weights to obtain the four Table V rows. It checks them
against the retained manuscript transcription at the displayed precision:
0.01 KiB and 0.001 ms. No observations are filtered by timing or rate gain.

CDF/tensor storage, process RSS and sampled workload increments are different
quantities. A zero sampled RSS increment is a recorded measurement, not a zero
total process-memory claim. Likewise, a full training checkpoint contains the
codec, frozen classifier, discriminator and configuration; its file size is
not the sender-state storage reported in Table V.

Original profiling records use CC BY 4.0 under `LICENSE-DATA.md`. Private
absolute paths and internal identifiers in the public text copies are mapped
in `data/neural/frozen_artifacts_manifest.json`; scientific numeric fields,
seeds and original fingerprints remain unchanged. Reaggregation uses archived
measurements; a new platform benchmark is an opt-in, separate operation.
