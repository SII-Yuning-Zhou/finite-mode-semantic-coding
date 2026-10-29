"""Deterministic task and target-distribution metrics."""

from __future__ import annotations

import hashlib

import numpy as np


DEFAULT_CLASS_LABELS = (0, 1, 2)
DEFAULT_MMD_BANDWIDTHS = (0.5, 1.0, 2.0, 4.0)


def _multiclass_log_loss(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_labels: tuple[int, ...],
) -> float:
    class_to_column = {label: index for index, label in enumerate(class_labels)}
    try:
        columns = np.asarray([class_to_column[int(label)] for label in labels])
    except KeyError as error:
        raise ValueError(f"Undeclared class label: {error.args[0]}") from error
    selected = probabilities[np.arange(len(labels)), columns]
    selected = np.clip(selected, np.finfo(np.float64).eps, 1.0)
    return float(-np.mean(np.log(selected)))


def _macro_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_labels: tuple[int, ...],
) -> float:
    values = []
    for label in class_labels:
        true_positive = np.sum((labels == label) & (predictions == label))
        false_positive = np.sum((labels != label) & (predictions == label))
        false_negative = np.sum((labels == label) & (predictions != label))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(values))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    """Top-label ECE with fixed equal-width confidence bins."""

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 2:
        raise ValueError("labels must be a vector and probabilities a matrix.")
    if len(labels) != len(probabilities) or len(labels) == 0:
        raise ValueError("labels and probabilities must have equal nonzero length.")
    if bins < 1:
        raise ValueError("bins must be positive.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and nonnegative.")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one.")

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(labels)), predictions]
    bin_index = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = bin_index == index
        if not np.any(selected):
            continue
        empirical_accuracy = np.mean(predictions[selected] == labels[selected])
        mean_confidence = np.mean(confidence[selected])
        error += float(np.mean(selected) * abs(empirical_accuracy - mean_confidence))
    return error


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_labels: tuple[int, ...] = DEFAULT_CLASS_LABELS,
    calibration_bins: int = 15,
) -> dict[str, float]:
    """Return ordinary, class-balanced, proper-loss, and calibration metrics."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), len(class_labels)):
        raise ValueError(
            "probabilities must have shape (number of labels, number of classes)."
        )
    predictions = probabilities.argmax(axis=1)
    present = [label for label in class_labels if np.any(labels == label)]
    if not present:
        raise ValueError("At least one declared class must occur in labels.")
    recalls = [
        float(np.mean(predictions[labels == label] == label)) for label in present
    ]
    balanced_accuracy = float(np.mean(recalls))
    accuracy = float(np.mean(predictions == labels))
    return {
        "cross_entropy": float(
            _multiclass_log_loss(labels, probabilities, class_labels)
        ),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": _macro_f1(labels, predictions, class_labels),
        "error": 1.0 - accuracy,
        "class_balanced_zero_one_risk": 1.0 - balanced_accuracy,
        "expected_calibration_error": expected_calibration_error(
            labels,
            probabilities,
            bins=calibration_bins,
        ),
    }


def grouped_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    class_labels: tuple[int, ...] = DEFAULT_CLASS_LABELS,
    calibration_bins: int = 15,
) -> dict[str, dict[str, float | int]]:
    """Classification metrics per physical identity or other fixed group."""

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    groups = np.asarray(groups).astype(str)
    if len(labels) != len(groups):
        raise ValueError("groups must have one value per label.")
    result: dict[str, dict[str, float | int]] = {}
    for group in sorted(np.unique(groups)):
        selected = groups == group
        metrics = classification_metrics(
            labels[selected],
            probabilities[selected],
            class_labels=class_labels,
            calibration_bins=calibration_bins,
        )
        result[group] = {
            "windows": int(np.sum(selected)),
            "class_index": int(labels[selected][0]),
            **metrics,
        }
    return result


def _stable_group_seed(seed: int, group: str) -> int:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def identity_balanced_indices(
    identities: np.ndarray,
    *,
    maximum_per_identity: int,
    seed: int,
) -> np.ndarray:
    """Choose a deterministic capped sample independently inside each identity."""

    identities = np.asarray(identities).astype(str)
    if identities.ndim != 1 or len(identities) == 0:
        raise ValueError("identities must be a nonempty vector.")
    if maximum_per_identity < 1:
        raise ValueError("maximum_per_identity must be positive.")
    selected: list[np.ndarray] = []
    for identity in sorted(np.unique(identities)):
        indices = np.flatnonzero(identities == identity)
        if len(indices) > maximum_per_identity:
            generator = np.random.Generator(
                np.random.PCG64(_stable_group_seed(seed, identity))
            )
            indices = np.sort(
                generator.choice(indices, size=maximum_per_identity, replace=False)
            )
        selected.append(indices)
    return np.concatenate(selected)


def mean_multi_kernel_rbf(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidths: tuple[float, ...] = DEFAULT_MMD_BANDWIDTHS,
    chunk_size: int = 512,
) -> float:
    """Mean of the multi-bandwidth RBF kernel without materializing NxM."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Kernel inputs must be matrices with equal width.")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Kernel inputs must be nonempty.")
    if chunk_size < 1 or not bandwidths or any(value <= 0 for value in bandwidths):
        raise ValueError("Kernel bandwidths and chunk_size must be positive.")
    total = 0.0
    for x_start in range(0, len(x), chunk_size):
        x_block = x[x_start : x_start + chunk_size]
        x_norm = np.sum(np.square(x_block), axis=1)[:, None]
        for y_start in range(0, len(y), chunk_size):
            y_block = y[y_start : y_start + chunk_size]
            squared_distance = (
                x_norm
                + np.sum(np.square(y_block), axis=1)[None, :]
                - 2.0 * x_block @ y_block.T
            )
            np.maximum(squared_distance, 0.0, out=squared_distance)
            block_kernel = np.zeros_like(squared_distance)
            for bandwidth in bandwidths:
                block_kernel += np.exp(
                    -squared_distance / (2.0 * bandwidth * bandwidth)
                )
            total += float(np.sum(block_kernel)) / len(bandwidths)
    return total / (len(x) * len(y))


def multi_kernel_mmd_squared(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidths: tuple[float, ...] = DEFAULT_MMD_BANDWIDTHS,
    chunk_size: int = 512,
    y_self_kernel_mean: float | None = None,
) -> float:
    """Biased nonnegative MMD^2 matching the training objective."""

    xx = mean_multi_kernel_rbf(
        x,
        x,
        bandwidths=bandwidths,
        chunk_size=chunk_size,
    )
    yy = (
        mean_multi_kernel_rbf(
            y,
            y,
            bandwidths=bandwidths,
            chunk_size=chunk_size,
        )
        if y_self_kernel_mean is None
        else float(y_self_kernel_mean)
    )
    xy = mean_multi_kernel_rbf(
        x,
        y,
        bandwidths=bandwidths,
        chunk_size=chunk_size,
    )
    return max(0.0, float(xx + yy - 2.0 * xy))


def sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    *,
    projections: int = 64,
    seed: int = 20260810,
) -> float:
    """Deterministic sliced 1-Wasserstein distance in feature space."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Sliced Wasserstein inputs must be matrices with equal width.")
    if len(x) == 0 or len(y) == 0 or projections < 1:
        raise ValueError("Sliced Wasserstein inputs and projections must be nonempty.")
    generator = np.random.Generator(np.random.PCG64(seed))
    directions = generator.normal(size=(projections, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    x_projection = x @ directions.T
    y_projection = y @ directions.T
    quantiles = np.linspace(0.0, 1.0, max(len(x), len(y)), endpoint=True)
    distances = []
    for column in range(projections):
        xq = np.quantile(x_projection[:, column], quantiles)
        yq = np.quantile(y_projection[:, column], quantiles)
        distances.append(np.mean(np.abs(xq - yq)))
    return float(np.mean(distances))


def spectral_distribution_metrics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidths: tuple[float, ...] = DEFAULT_MMD_BANDWIDTHS,
    projections: int = 64,
    seed: int = 20260810,
    y_self_kernel_mean: float | None = None,
) -> dict[str, float]:
    return {
        "multi_kernel_mmd_squared": multi_kernel_mmd_squared(
            x,
            y,
            bandwidths=bandwidths,
            y_self_kernel_mean=y_self_kernel_mean,
        ),
        "sliced_wasserstein": sliced_wasserstein(
            x,
            y,
            projections=projections,
            seed=seed,
        ),
    }
