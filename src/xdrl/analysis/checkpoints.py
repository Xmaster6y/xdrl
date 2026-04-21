from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint file as a dictionary."""
    return torch.load(Path(path), map_location=map_location)


def checkpoint_keys(checkpoint: Mapping[str, Any]) -> list[str]:
    """Return sorted top-level keys available in a checkpoint."""
    return sorted(checkpoint.keys())


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    """Summarize a tensor with simple scalar statistics."""
    detached = tensor.detach().float()
    return {
        "mean": detached.mean().item(),
        "std": detached.std(unbiased=False).item(),
        "min": detached.min().item(),
        "max": detached.max().item(),
    }
