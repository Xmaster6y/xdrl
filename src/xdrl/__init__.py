from importlib.metadata import PackageNotFoundError, version

from xdrl.interactions import (
    AgentSelector,
    InteractionDescriptor,
    InteractionPhase,
    InteractionTopology,
    MultiAgentSemantics,
    RecurrentCollectorMode,
    RecurrentSemantics,
    RecurrentStateTransition,
    RuntimeInteractionContext,
    SemanticTarget,
)
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
from xdrl.observations import (
    DimensionReduction,
    ObservationKind,
    ObservationRecord,
    ObservationTrace,
    ReductionKind,
    RetentionPolicy,
    TensorRetention,
)
from xdrl.tdhook import TDHookInteractionAdapter
from xdrl.types import ModelRole, TensorDictSchema

try:
    __version__ = version("xdrl")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "InteractionDescriptor",
    "InteractionTopology",
    "Intervention",
    "InterventionController",
    "InterventionRecord",
    "InterventionScope",
    "InterventionTarget",
    "InterventionTiming",
    "InterventionValidationError",
    "InteractionPhase",
    "AgentSelector",
    "MultiAgentSemantics",
    "ModelRole",
    "ObservationKind",
    "ObservationRecord",
    "ObservationTrace",
    "DimensionReduction",
    "ReductionKind",
    "RecurrentCollectorMode",
    "RecurrentSemantics",
    "RecurrentStateTransition",
    "RetentionPolicy",
    "RuntimeInteractionContext",
    "SemanticTarget",
    "PairedInterventionResult",
    "TDHookInteractionAdapter",
    "TDHookInterventionFactory",
    "TensorDictSchema",
    "TensorRetention",
    "run_paired",
]
