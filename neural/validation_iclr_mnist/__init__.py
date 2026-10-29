"""ICLR-aligned validation for mode-limited target semantic coding.

The package deliberately separates three claims:

* ``bernoulli`` checks the proved finite-mode boundary by Monte Carlo;
* ``mnist`` recovers the neural validation carrier used by the ICLR paper;
* ``finite_dither`` maps the paper's deployable resources ``K`` and ``H(V)``
  to a frozen finite shared-dither codebook.

Neural latent bits are an operational upper bound for generating the decoded
reconstruction.  They are not identified with the theorem's ``H(Y|V)``.
"""

from .config import ExperimentConfig, load_experiment_config

__all__ = ["ExperimentConfig", "load_experiment_config"]
