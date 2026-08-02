from importlib.metadata import PackageNotFoundError, version

from xdrl.interactions import InteractionDescriptor, InteractionPhase, RuntimeInteractionContext
from xdrl.observations import ObservationKind, ObservationRecord, ObservationTrace, RetentionPolicy, TensorRetention
from xdrl.tdhook import TDHookInteractionAdapter
from xdrl.types import ModelRole, TensorDictSchema

try:
    __version__ = version("xdrl")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "InteractionDescriptor",
    "InteractionPhase",
    "ModelRole",
    "ObservationKind",
    "ObservationRecord",
    "ObservationTrace",
    "RetentionPolicy",
    "RuntimeInteractionContext",
    "TDHookInteractionAdapter",
    "TensorDictSchema",
    "TensorRetention",
]
