"""Deterministic runtime, mode scheduling, and atomic artifact helpers."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from validation_semcom.schedule import DyadicSchedule

from .config import ExperimentConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def mode_schedule(config: ExperimentConfig) -> DyadicSchedule:
    return DyadicSchedule(config.dither.counts)


def draw_modes(
    config: ExperimentConfig,
    count: int,
    *,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    if config.dither.kind == "continuous":
        return torch.zeros(count, dtype=torch.long, device=device)
    modes, _ = mode_schedule(config).draw(
        count, seed=config.dither.schedule_seed + 1000003 * offset
    )
    return torch.from_numpy(modes).to(device=device, dtype=torch.long)


def dither_generator(
    config: ExperimentConfig, *, offset: int, device: torch.device
) -> torch.Generator | None:
    if config.dither.kind != "continuous":
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(config.dither.schedule_seed + 1000003 * offset)
    return generator


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
