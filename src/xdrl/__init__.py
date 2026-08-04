from importlib.metadata import PackageNotFoundError, version

from xdrl.compatibility import (
    ADAPTER_CONFORMANCE,
    PRIVATE_UPSTREAM_APIS,
    SUPPORT_DEFINITIONS,
    SUPPORTED_DEPENDENCIES,
    SUPPORTED_PYTHON,
    CompatibilityBoundaryError,
    ConformanceCheck,
    ConformanceSuite,
    PrivateAPIUsage,
    SupportLevel,
    VersionRequirement,
    installed_dependency_versions,
    validate_runtime_compatibility,
)
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
from xdrl.provenance import PROVENANCE_SCHEMA_REVISION, ProvenanceManifest, ProvenanceSchemaError
from xdrl.tdhook import TDHookInteractionAdapter
from xdrl.types import ModelRole, TensorDictSchema

try:
    __version__ = version("xdrl")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "ADAPTER_CONFORMANCE",
    "CompatibilityBoundaryError",
    "ConformanceCheck",
    "ConformanceSuite",
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
    "PRIVATE_UPSTREAM_APIS",
    "PrivateAPIUsage",
    "PROVENANCE_SCHEMA_REVISION",
    "ProvenanceManifest",
    "ProvenanceSchemaError",
    "DimensionReduction",
    "ReductionKind",
    "RecurrentCollectorMode",
    "RecurrentSemantics",
    "RecurrentStateTransition",
    "RetentionPolicy",
    "RuntimeInteractionContext",
    "SemanticTarget",
    "SUPPORTED_DEPENDENCIES",
    "SUPPORTED_PYTHON",
    "SUPPORT_DEFINITIONS",
    "SupportLevel",
    "PairedInterventionResult",
    "TDHookInteractionAdapter",
    "TDHookInterventionFactory",
    "TensorDictSchema",
    "TensorRetention",
    "VersionRequirement",
    "installed_dependency_versions",
    "run_paired",
    "validate_runtime_compatibility",
]
