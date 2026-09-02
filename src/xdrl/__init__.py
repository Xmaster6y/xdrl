"""Typed TorchRL interaction boundaries with native TDHook workflows."""

from importlib.metadata import PackageNotFoundError, version

from xdrl.interactions import Interaction, InteractionSpec, RecurrentSemantics, RecurrentStateTransition
from xdrl.tdhook import run_workflow
from xdrl.types import (
    BatchSemantics,
    KeyRole,
    KeySchema,
    ModelRole,
    SchemaValidationError,
    TensorDictKey,
    TensorDictSchema,
)

try:
    __version__ = version("xdrl")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "BatchSemantics",
    "Interaction",
    "InteractionSpec",
    "KeyRole",
    "KeySchema",
    "ModelRole",
    "RecurrentSemantics",
    "RecurrentStateTransition",
    "SchemaValidationError",
    "TensorDictKey",
    "TensorDictSchema",
    "run_workflow",
]
