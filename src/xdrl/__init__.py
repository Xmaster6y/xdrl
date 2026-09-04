"""Interpretability extensions for native TorchRL objects."""

from importlib.metadata import PackageNotFoundError, version

from xdrl.interpretation import Component, RecurrentSemantics, RecurrentStateTransition, interpret

try:
    __version__ = version("xdrl")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "Component",
    "RecurrentSemantics",
    "RecurrentStateTransition",
    "interpret",
]
