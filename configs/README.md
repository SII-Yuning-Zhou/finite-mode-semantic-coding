# Experiment configurations

The bundled configurations describe the published MNIST operating points,
classifier preparation, model architecture and training settings.

The public interface selects the appropriate defaults:

```bash
python mnist.py evaluate --rate-weight 5000 --device cuda:0
python mnist.py profile --rate-weight 5000 --device cpu
python mnist.py train --rate-weight 5000 --device cuda:0
```

Use `--grid FILE` or `--config FILE` where supported to select a custom
configuration. New training writes its derived configurations under the
requested `runs/` directory.

See the [MNIST guide](../docs/NEURAL_REPRODUCTION.md) for model, split, precision
and optimizer settings. The [archive mapping](../docs/ARCHIVE_COMPATIBILITY.md)
records the original configuration identifiers and coding-protocol settings.
