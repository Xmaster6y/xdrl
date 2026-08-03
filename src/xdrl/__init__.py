from importlib.metadata import PackageNotFoundError, version

from xdrl.interactions import InteractionDescriptor, InteractionPhase, RuntimeInteractionContext
from xdrl.interventions import (
    Intervention,
    InterventionController,
    InterventionRecord,
    InterventionScope,
    InterventionTarget,
    InterventionTiming,
    InterventionValidationError,
    PairedInterventionResult,
    TDHookInterventionFactory,
    run_paired,
)
from xdrl.observations import ObservationKind, ObservationRecord, ObservationTrace, RetentionPolicy, TensorRetention
from xdrl.tdhook import TDHookInteractionAdapter
from xdrl.types import ModelRole, TensorDictSchema

try:
    __version__ = version("xdrl")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "InteractionDescriptor",
    "Intervention",
    "InterventionController",
    "InterventionRecord",
    "InterventionScope",
    "InterventionTarget",
    "InterventionTiming",
    "InterventionValidationError",
    "InteractionPhase",
    "ModelRole",
    "ObservationKind",
    "ObservationRecord",
    "ObservationTrace",
    "RetentionPolicy",
    "RuntimeInteractionContext",
    "PairedInterventionResult",
    "TDHookInteractionAdapter",
    "TDHookInterventionFactory",
    "TensorDictSchema",
    "TensorRetention",
    "run_paired",
]
