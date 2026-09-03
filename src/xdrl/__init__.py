"""TorchRL interaction semantics with native TDHook workflows."""

from importlib.metadata import PackageNotFoundError, version

from xdrl.interactions import Interaction, RecurrentSemantics, RecurrentStateTransition
from xdrl.tdhook import run_workflow

try:
    __version__ = version("xdrl")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "Interaction",
    "RecurrentSemantics",
    "RecurrentStateTransition",
    "run_workflow",
]
