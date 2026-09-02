"""Deterministic identities for reproduction inputs and model state."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

import torch


def bytes_digest(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash a tensor's dtype, shape, and contiguous CPU values."""
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return bytes_digest(header + value.numpy().tobytes())


def named_tensor_digest(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Hash named tensors independent of mapping iteration order."""
    manifest = [
        {"name": name, "sha256": tensor_digest(tensor)}
        for name, tensor in sorted(named_tensors, key=lambda item: item[0])
    ]
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return bytes_digest(payload)


def module_digest(module: torch.nn.Module) -> str:
    """Hash a module's named parameters and buffers deterministically."""
    return named_tensor_digest(module.state_dict().items())


def repository_revision(path: str | Path | None = None) -> str:
    """Return the Git revision, suffixed with ``+dirty`` when tracked files differ."""
    cwd = Path(path) if path is not None else None
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"{revision}+dirty" if dirty else revision


__all__ = ["bytes_digest", "module_digest", "named_tensor_digest", "repository_revision", "tensor_digest"]
