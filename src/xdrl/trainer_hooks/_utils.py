from __future__ import annotations

import torch


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
