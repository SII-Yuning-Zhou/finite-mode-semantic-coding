"""Health Gate for the frozen MNIST task classifier.

The task head is trained on clean target-domain MNIST. Accuracy on the
7x7-degraded source is useful as an input baseline, but it must not be a
health requirement: making the classifier invariant to that degradation
would partly perform the restoration task that the codec is meant to learn.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


CLEAN_ACCURACY_MINIMUM = 0.98


def classifier_health_failures(
    diagnostics: Mapping[str, float | int],
) -> list[str]:
    """Return hard health failures without gating the degraded baseline."""

    failures: list[str] = []
    for name in ("clean_accuracy", "degraded_7x7_accuracy"):
        try:
            value = float(diagnostics[name])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{name} is missing or non-numeric")
            continue
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            failures.append(f"{name}={value!r} is outside [0,1]")
    clean = diagnostics.get("clean_accuracy")
    if isinstance(clean, (float, int)) and math.isfinite(float(clean)):
        if float(clean) < CLEAN_ACCURACY_MINIMUM:
            failures.append(
                f"clean_accuracy={float(clean):.6f} < {CLEAN_ACCURACY_MINIMUM:.2f}"
            )
    return failures


def classifier_gate_rule() -> dict[str, object]:
    return {
        "clean_accuracy_minimum": CLEAN_ACCURACY_MINIMUM,
        "degraded_7x7_accuracy_role": "reported_input_baseline_only_not_gated",
        "reason": (
            "The frozen task head is a clean target-domain evaluator; requiring "
            "degraded-input invariance would transfer restoration work from the "
            "codec to the evaluator."
        ),
    }
