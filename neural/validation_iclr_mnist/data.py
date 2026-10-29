"""MNIST 4x super-resolution data used by the ICLR validation carrier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset


def degrade_mnist(hr: torch.Tensor) -> torch.Tensor:
    """Bilinear 28x28 -> 7x7 -> 28x28 degradation from the ICLR experiment."""

    original_dimensions = hr.ndim
    if original_dimensions == 3:
        hr = hr.unsqueeze(0)
    if hr.ndim != 4 or hr.shape[-2:] != (28, 28):
        raise ValueError("MNIST tensors must have shape Cx28x28 or BxCx28x28.")
    low = F.interpolate(
        hr, size=(7, 7), mode="bilinear", align_corners=False, antialias=True
    )
    restored = F.interpolate(
        low, size=(28, 28), mode="bilinear", align_corners=False
    )
    return restored.squeeze(0) if original_dimensions == 3 else restored


class MNISTSourceDataset(Dataset):
    """Return degraded input, paired HR reference, label, and immutable index."""

    def __init__(self, base: Dataset, indices: np.ndarray | list[int]):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor | int]:
        index = int(self.indices[position])
        hr, label = self.base[index]
        if not isinstance(hr, torch.Tensor):
            raise TypeError("MNIST base dataset must apply torchvision.transforms.ToTensor.")
        return {
            "x": degrade_mnist(hr),
            "hr": hr,
            "label": int(label),
            "index": index,
        }


class MNISTTargetDataset(Dataset):
    """Independent clean target-distribution pool for unpaired WGAN batches."""

    def __init__(self, base: Dataset, indices: np.ndarray | list[int]):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor | int]:
        index = int(self.indices[position])
        hr, label = self.base[index]
        return {"hr": hr, "label": int(label), "index": index}


@dataclass(frozen=True)
class MNISTSplits:
    source_train: MNISTSourceDataset
    target_train: MNISTTargetDataset
    validation: MNISTSourceDataset
    test: MNISTSourceDataset | None


def fixed_train_validation_indices(
    size: int, *, validation_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < validation_size < size:
        raise ValueError("validation_size must be strictly between zero and size.")
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(size)
    validation = np.sort(permutation[:validation_size])
    train = np.sort(permutation[validation_size:])
    return train, validation


def load_mnist_splits(
    root: str | Path,
    *,
    validation_size: int,
    seed: int,
    download: bool,
    include_official_test: bool = False,
) -> MNISTSplits:
    """Load MNIST lazily so S0 remains usable without PyTorch/torchvision."""

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "MNIST validation requires torchvision compatible with the installed torch."
        ) from exc
    transform = transforms.ToTensor()
    root = Path(root)
    train_base = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    train_indices, validation_indices = fixed_train_validation_indices(
        len(train_base), validation_size=validation_size, seed=seed
    )
    # The source and clean-target loaders shuffle independently.  Equal pools
    # are intentional; the training loop prevents index-wise pairing.
    test_dataset = None
    if include_official_test:
        test_base = datasets.MNIST(
            root=root, train=False, transform=transform, download=download
        )
        test_dataset = MNISTSourceDataset(test_base, np.arange(len(test_base)))
    return MNISTSplits(
        source_train=MNISTSourceDataset(train_base, train_indices),
        target_train=MNISTTargetDataset(train_base, train_indices),
        validation=MNISTSourceDataset(train_base, validation_indices),
        test=test_dataset,
    )


def paired_clean_subset(source: MNISTSourceDataset) -> Subset:
    """Expose source entries for clean-classifier pretraining only."""

    return Subset(source, list(range(len(source))))
