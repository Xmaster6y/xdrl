from __future__ import annotations

from typing import Any

from torch import nn


def freeze_module(module: nn.Module) -> nn.Module:
    """Disable gradients for all module parameters."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def unfreeze_module(module: nn.Module) -> nn.Module:
    """Enable gradients for all module parameters."""
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    return module


def set_eval_mode(module: nn.Module) -> nn.Module:
    """Set module to eval mode and return it."""
    module.eval()
    return module


def set_train_mode(module: nn.Module) -> nn.Module:
    """Set module to train mode and return it."""
    module.train()
    return module


def copy_policy_weights(source: nn.Module, target: nn.Module, strict: bool = True) -> tuple[list[str], list[str]]:
    """Copy source parameters to target and report missing and unexpected keys."""
    incompatible: Any = target.load_state_dict(source.state_dict(), strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
