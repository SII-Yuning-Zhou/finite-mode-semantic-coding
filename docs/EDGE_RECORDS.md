# Endpoint profile records

The eight original profiles under [results/](../results/) cover four rate
weights (0, 200, 5,000 and 50,000) on CPU and CUDA. Each profile contains five
offset levels (1, 3, 7, 15 and 31), with marginal and conditional coding paths.
Together they contain 80 ledger rows and 160 encode/decode timing distributions.

The [artifact manifest](../data/neural/frozen_artifacts_manifest.json) identifies
the exact source files and their byte fingerprints.

## Fields and units

| Field | Meaning and unit |
|---|---|
| `rate_weight`, `seed` | Objective parameter and fixed codec seed |
| `checkpoint_sha256` | Identity of the original checkpoint bytes |
| `hardware` | Recorded device, platform, CPU counts and GPU metadata |
| `measurement_protocol` | Batch, threads, precision, warm-up and sampling settings |
| `model_resources` | Parameter counts, tensor-state bytes and convolution MACs; component scope is named in the field |
| `levels[].levels` | Offset cardinality L |
| `conditional_table_bytes`, `marginal_table_bytes` | Stored CDF bytes; divide by 1024 for KiB |
| `levels[].paths.*.encode`, `.decode` | Per-image timing distributions |
| `raw_seconds` | Original timed-call durations, in seconds |
| `mean_ms`, `median_ms`, `p95_ms` | Timing summaries, in milliseconds |
| `sender_memory`, `receiver_memory` | Loaded-process RSS, sampled workload increments and CUDA allocation statistics, in bytes |

## Measurement protocol

Each endpoint distribution has 100 timing calls after 20 warm-up calls:
16,000 timed calls in total. The protocol uses batch one, 128 validation images,
one intra-op and inter-op CPU thread, FP32, AMP off, TF32 off and a 32-bit frame.
Setup is excluded.

Sender timing includes neural encoding, shared scheduling, quantization,
indexed rANS and framing. Receiver timing includes rANS, neural decoding and
output transfer. The profiling implementation defines synchronization and
memory sampling.

The recorded host is x86_64 with 128 logical and 64 physical CPUs; its exact
CPU model name was not stored. GPU metadata identify NVIDIA GeForce RTX 4090.

## Rebuild the resource ledger

```bash
python -m reproduce.tables --output outputs/tables
```

The command rebuilds all 80 rows, then selects the conditional path at L=1
and L=31. For each CPU/CUDA and L pair, it takes the minimum/maximum mean and
p95 across all four rate weights to obtain the four Table V rows. The
comparison with the retained manuscript transcription uses its displayed
precision: 0.01 KiB and 0.001 ms.

CDF/tensor storage, process RSS and workload increments are separate quantities.
A zero sampled RSS increment is a recorded increment, not the total process
footprint. A full training checkpoint also contains classifier, discriminator
and configuration state; checkpoint file size is distinct from endpoint
sender-state storage.

## Collect new measurements

```bash
python mnist.py profile --rate-weight 5000 --device cpu --threads 1 --output-dir outputs/edge_profile
```

Full options are in [MNIST codec](NEURAL_REPRODUCTION.md). Reaggregation uses
the saved timing observations; profiling measures the reader's own machine.

These author-generated records use CC BY 4.0. Source fingerprints and numeric
fields are preserved; public text-path transformations are recorded in the
artifact manifest.
