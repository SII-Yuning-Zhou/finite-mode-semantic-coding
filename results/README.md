# Models and endpoint measurements

This directory contains the four released MNIST codec checkpoints, the frozen
grayscale classifier, and the original CPU/CUDA endpoint profiles.

```bash
python mnist.py list-models
python mnist.py verify
```

Codec objective weights: **0, 200, 5,000, and 50,000**. Model identities and
recorded measurements are selected automatically by the public interface.

- [Model evaluation and profiling](../docs/NEURAL_REPRODUCTION.md)
- [Endpoint fields and units](../docs/EDGE_RECORDS.md)
- [Artifact identity manifest](../data/neural/frozen_artifacts_manifest.json)
- [Versioned models and data](https://doi.org/10.5281/zenodo.23774113)

Derived measurements go to `outputs/` and new training runs to `runs/`.
The bundled model files and observations remain the inputs for reproduction.
