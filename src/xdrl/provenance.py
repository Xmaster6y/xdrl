"""Versioned, serialisable evidence for TDHook workflows in XDRL."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import MISSING, asdict, dataclass, field, fields
from enum import Enum
from types import MappingProxyType
from typing import Any

from packaging.version import InvalidVersion, Version
from tdhook.workflow import CompatibilityDecision, PlannedExecution, WorkflowPlan

from xdrl.compatibility import installed_dependency_versions
from xdrl.interactions import (
    AgentSelector,
    InteractionTopology,
    InteractionContract,
    InteractionPhase,
    LifecycleEvent,
    LifecycleEventType,
    MultiAgentSemantics,
    RecurrentCollectorMode,
    RecurrentSemantics,
    RecurrentStateTransition,
    SemanticTarget,
)
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema

WORKFLOW_PROVENANCE_SCHEMA_REVISION = 3


class ProvenanceSchemaError(ValueError):
    """A workflow provenance payload violates the current schema."""


class ArtifactDigestAlgorithm(str, Enum):
    """Supported cryptographic digest algorithms for external artifacts."""

    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"


class InputArtifactRole(str, Enum):
    """The declared role an external input plays in a workflow run."""

    MODEL_CHECKPOINT = "model_checkpoint"
    DATASET = "dataset"
    ENVIRONMENT_BUILD = "environment_build"
    EVALUATION_SPLIT = "evaluation_split"
    LEVEL_SET = "level_set"
    REFERENCE_RESULT = "reference_result"
    OTHER = "other"


class OutputArtifactRole(str, Enum):
    """The declared role of an artifact emitted by a workflow run."""

    METRICS_BUNDLE = "metrics_bundle"
    OBSERVATION_TRACE = "observation_trace"
    ATTRIBUTION = "attribution"
    INTERVENTION_RESULT = "intervention_result"
    FIGURE = "figure"
    OTHER = "other"


_DIGEST_HEX_LENGTH = {
    ArtifactDigestAlgorithm.SHA256: 64,
    ArtifactDigestAlgorithm.SHA384: 96,
    ArtifactDigestAlgorithm.SHA512: 128,
    ArtifactDigestAlgorithm.BLAKE2B: 128,
    ArtifactDigestAlgorithm.BLAKE2S: 64,
}


@dataclass(frozen=True, slots=True)
class InputArtifactReference:
    """Tensor-free identity for one external input used by a workflow."""

    identity: str
    role: InputArtifactRole
    digest_algorithm: ArtifactDigestAlgorithm
    digest_value: str
    source: str | None = None
    revision: str | None = None
    source_is_immutable: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_artifact_reference(self, "input artifact")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""
        return _artifact_reference_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InputArtifactReference:
        """Strictly decode one input artifact reference."""
        values = _artifact_reference_values(payload, "input artifact", InputArtifactRole)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class OutputArtifactReference:
    """Tensor-free identity for one result artifact emitted by a workflow."""

    identity: str
    role: OutputArtifactRole
    digest_algorithm: ArtifactDigestAlgorithm
    digest_value: str
    source: str | None = None
    revision: str | None = None
    source_is_immutable: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_artifact_reference(self, "output artifact")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""
        return _artifact_reference_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OutputArtifactReference:
        """Strictly decode one output artifact reference."""
        values = _artifact_reference_values(payload, "output artifact", OutputArtifactRole)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class WorkflowExecutionEvidence:
    """Serializable projection of one public TDHook planned execution."""

    steps: tuple[str, ...]
    kind: str
    in_keys: tuple[tuple[str, ...], ...]
    out_keys: tuple[tuple[str, ...], ...]
    model_passes: int
    gradient_mode: str | None
    coexecuted: bool

    @classmethod
    def from_execution(cls, execution: PlannedExecution) -> WorkflowExecutionEvidence:
        """Capture only public, stable fields from a TDHook execution."""
        return cls(
            steps=tuple(execution.steps),
            kind=execution.kind,
            in_keys=tuple(_key_path(key) for key in execution.in_keys),
            out_keys=tuple(_key_path(key) for key in execution.out_keys),
            model_passes=execution.model_passes,
            gradient_mode=execution.gradient_mode.value if execution.gradient_mode is not None else None,
            coexecuted=execution.coexecuted,
        )


@dataclass(frozen=True, slots=True)
class WorkflowCompatibilityEvidence:
    """Serializable projection of one public TDHook compatibility decision."""

    existing_steps: tuple[str, ...]
    candidate_step: str
    compatible: bool
    reason: str

    @classmethod
    def from_decision(cls, decision: CompatibilityDecision) -> WorkflowCompatibilityEvidence:
        """Capture one TDHook planning decision without private method state."""
        return cls(
            existing_steps=tuple(decision.existing_steps),
            candidate_step=decision.candidate_step,
            compatible=decision.compatible,
            reason=decision.reason,
        )


@dataclass(frozen=True, slots=True)
class WorkflowPlanEvidence:
    """Immutable serialisable evidence derived from a public TDHook plan."""

    executions: tuple[WorkflowExecutionEvidence, ...]
    compatibility: tuple[WorkflowCompatibilityEvidence, ...]

    @classmethod
    def from_plan(cls, plan: WorkflowPlan) -> WorkflowPlanEvidence:
        """Snapshot a TDHook plan without retaining methods or hook programs."""
        return cls(
            executions=tuple(WorkflowExecutionEvidence.from_execution(item) for item in plan.executions),
            compatibility=tuple(WorkflowCompatibilityEvidence.from_decision(item) for item in plan.compatibility),
        )

    @property
    def model_passes(self) -> int:
        """Return the model passes declared across all planned executions."""
        return sum(execution.model_passes for execution in self.executions)


@dataclass(frozen=True, slots=True)
class WorkflowProvenance:
    """Reproduction evidence for one successful typed TDHook workflow run."""

    schema_revision: int
    interaction_contract: Mapping[str, Any]
    workflow_plan: WorkflowPlanEvidence
    configured_steps: tuple[str, ...]
    lifecycle_events: tuple[LifecycleEvent, ...]
    dependencies: Mapping[str, str]
    code_revision: str
    input_artifacts: tuple[InputArtifactReference, ...]
    output_artifacts: tuple[OutputArtifactReference, ...]
    seed: int | None = None

    def __post_init__(self) -> None:
        if type(self.schema_revision) is not int or self.schema_revision != WORKFLOW_PROVENANCE_SCHEMA_REVISION:
            raise ProvenanceSchemaError(
                f"unsupported workflow provenance schema revision {self.schema_revision}; "
                f"expected {WORKFLOW_PROVENANCE_SCHEMA_REVISION}"
            )
        contract = _contract_projection(_thaw_json(self.interaction_contract))
        interaction_id = contract["identity"]
        _nonempty_string(self.code_revision, "code_revision")
        if self.seed is not None and type(self.seed) is not int:
            raise ProvenanceSchemaError("seed must be an integer or null")
        dependencies = _validate_dependencies(self.dependencies)
        configured_steps = _configured_step_sequence(self.configured_steps)
        input_artifacts, output_artifacts = _artifact_references(self.input_artifacts, self.output_artifacts)
        if any(event.interaction_id != interaction_id for event in self.lifecycle_events):
            raise ProvenanceSchemaError("lifecycle events must belong to the interaction contract")
        if self.model_calls != self.workflow_plan.model_passes:
            raise ProvenanceSchemaError(
                "workflow plan and lifecycle evidence disagree: "
                f"plan declares {self.workflow_plan.model_passes}, events contain {self.model_calls} successful calls"
            )
        _validate_lifecycle(self.lifecycle_events, contract, self.workflow_plan.model_passes)
        object.__setattr__(self, "interaction_contract", _freeze_json(contract))
        object.__setattr__(self, "configured_steps", configured_steps)
        object.__setattr__(self, "dependencies", MappingProxyType(dependencies))
        object.__setattr__(self, "input_artifacts", input_artifacts)
        object.__setattr__(self, "output_artifacts", output_artifacts)
        try:
            json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ProvenanceSchemaError(f"workflow provenance contains a non-serialisable value: {error}") from error

    @classmethod
    def validate_run_metadata(
        cls,
        *,
        code_revision: str,
        seed: int | None = None,
        dependencies: Mapping[str, str] | None = None,
        input_artifacts: tuple[InputArtifactReference, ...] = (),
        output_artifacts: tuple[OutputArtifactReference, ...] = (),
    ) -> dict[str, str]:
        """Validate reproduction metadata before a workflow mutates live state."""
        _nonempty_string(code_revision, "code_revision")
        if seed is not None and type(seed) is not int:
            raise ProvenanceSchemaError("seed must be an integer or null")
        resolved = installed_dependency_versions() if dependencies is None else dependencies
        _artifact_references(input_artifacts, output_artifacts)
        return _validate_dependencies(dict(resolved))

    @classmethod
    def capture(
        cls,
        contract: InteractionContract,
        plan: WorkflowPlan,
        configured_steps: tuple[str, ...],
        events: tuple[LifecycleEvent, ...],
        *,
        code_revision: str,
        seed: int | None = None,
        dependencies: Mapping[str, str] | None = None,
        input_artifacts: tuple[InputArtifactReference, ...] = (),
        output_artifacts: tuple[OutputArtifactReference, ...] = (),
    ) -> WorkflowProvenance:
        """Capture verified workflow evidence from public XDRL and TDHook state."""
        validated_dependencies = cls.validate_run_metadata(
            code_revision=code_revision,
            seed=seed,
            dependencies=dependencies,
            input_artifacts=input_artifacts,
            output_artifacts=output_artifacts,
        )
        return cls(
            schema_revision=WORKFLOW_PROVENANCE_SCHEMA_REVISION,
            interaction_contract=_json_copy(contract.to_dict(), "interaction_contract"),
            workflow_plan=WorkflowPlanEvidence.from_plan(plan),
            configured_steps=_configured_step_sequence(configured_steps),
            lifecycle_events=events,
            dependencies=validated_dependencies,
            code_revision=code_revision,
            input_artifacts=input_artifacts,
            output_artifacts=output_artifacts,
            seed=seed,
        )

    @property
    def interaction_id(self) -> str:
        """Return the stable interaction identity recorded by the contract."""
        return str(self.interaction_contract["identity"])

    @property
    def model_calls(self) -> int:
        """Return successful root calls observed by XDRL."""
        return sum(event.kind is LifecycleEventType.AFTER for event in self.lifecycle_events)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible payload."""
        payload = {
            "schema_revision": self.schema_revision,
            "interaction_contract": _thaw_json(self.interaction_contract),
            "workflow_plan": asdict(self.workflow_plan),
            "configured_steps": list(self.configured_steps),
            "lifecycle_events": [event.to_dict() for event in self.lifecycle_events],
            "dependencies": dict(self.dependencies),
            "code_revision": self.code_revision,
            "input_artifacts": [artifact.to_dict() for artifact in self.input_artifacts],
            "output_artifacts": [artifact.to_dict() for artifact in self.output_artifacts],
            "seed": self.seed,
        }
        return _json_copy(payload, "workflow provenance")

    def to_json(self) -> str:
        """Encode workflow provenance deterministically."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkflowProvenance:
        """Decode provenance while rejecting unknown revisions or fields."""
        if not isinstance(payload, Mapping):
            raise ProvenanceSchemaError("workflow provenance must contain an object")
        revision = payload.get("schema_revision")
        if type(revision) is not int or revision != WORKFLOW_PROVENANCE_SCHEMA_REVISION:
            if revision == 2:
                raise ProvenanceSchemaError(
                    "workflow provenance schema revision 2 predates artifact references and cannot be migrated "
                    "without caller-supplied artifact identities; decode it with an XDRL version that supports revision 2"
                )
            raise ProvenanceSchemaError(
                f"unsupported workflow provenance schema revision {revision!r}; "
                f"expected {WORKFLOW_PROVENANCE_SCHEMA_REVISION}"
            )
        model_fields = fields(cls)
        known = {item.name for item in model_fields}
        required = {item.name for item in model_fields if item.default is MISSING and item.default_factory is MISSING}
        missing = required - set(payload)
        unknown = set(payload) - known
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing fields: {', '.join(sorted(missing))}")
            if unknown:
                details.append(f"unknown fields: {', '.join(sorted(unknown))}")
            raise ProvenanceSchemaError("; ".join(details))
        contract = _contract_projection(payload["interaction_contract"])
        plan = _plan_evidence(payload["workflow_plan"])
        configured_steps = _configured_steps(payload["configured_steps"])
        events = _lifecycle_events(payload["lifecycle_events"])
        dependencies = _validate_dependencies(_string_mapping(payload["dependencies"], "dependencies"))
        input_artifacts = _decoded_artifacts(payload["input_artifacts"], "input_artifacts", InputArtifactReference)
        output_artifacts = _decoded_artifacts(payload["output_artifacts"], "output_artifacts", OutputArtifactReference)
        seed = payload.get("seed")
        if seed is not None and type(seed) is not int:
            raise ProvenanceSchemaError("seed must be an integer or null")
        return cls(
            schema_revision=revision,
            interaction_contract=contract,
            workflow_plan=plan,
            configured_steps=configured_steps,
            lifecycle_events=events,
            dependencies=dependencies,
            code_revision=_nonempty_string(payload["code_revision"], "code_revision"),
            input_artifacts=input_artifacts,
            output_artifacts=output_artifacts,
            seed=seed,
        )

    @classmethod
    def from_json(cls, payload: str) -> WorkflowProvenance:
        """Decode workflow provenance from JSON."""
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProvenanceSchemaError(f"invalid provenance JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise ProvenanceSchemaError("workflow provenance JSON must contain an object")
        return cls.from_dict(value)


_CONTRACT_FIELDS = {
    "identity",
    "role",
    "phase",
    "module_path",
    "input_schema",
    "output_schema",
    "batch_dimensions",
    "environment",
    "time_dimension",
    "agent_dimension",
    "objective",
    "exploration_mode",
    "gradient_enabled",
    "inference_mode",
    "autocast_device_type",
    "autocast_enabled",
    "logical_step",
    "episode_id",
    "trajectory_id",
    "model_id",
    "checkpoint_id",
    "module_training",
    "recurrent",
    "multi_agent",
}


def _contract_projection(value: Any) -> dict[str, Any]:
    field = "interaction_contract"
    entry = _exact_mapping(value, field, _CONTRACT_FIELDS)
    contract = dict(entry)
    contract["identity"] = _nonempty_string(entry["identity"], f"{field}.identity")
    contract["role"] = _enum_string(entry["role"], ModelRole, f"{field}.role")
    contract["phase"] = _enum_string(entry["phase"], InteractionPhase, f"{field}.phase")
    contract["module_path"] = _nonempty_string(entry["module_path"], f"{field}.module_path")
    contract["input_schema"] = _schema_projection(entry["input_schema"], f"{field}.input_schema")
    contract["output_schema"] = _schema_projection(entry["output_schema"], f"{field}.output_schema")
    contract["batch_dimensions"] = list(_strings(entry["batch_dimensions"], f"{field}.batch_dimensions"))
    for name in (
        "environment",
        "time_dimension",
        "agent_dimension",
        "objective",
        "exploration_mode",
        "autocast_device_type",
        "model_id",
        "checkpoint_id",
    ):
        contract[name] = _optional_string(entry[name], f"{field}.{name}")
    for name in ("gradient_enabled", "inference_mode", "autocast_enabled"):
        contract[name] = _boolean(entry[name], f"{field}.{name}")
    contract["module_training"] = _optional_boolean(entry["module_training"], f"{field}.module_training")
    contract["logical_step"] = _optional_integer(entry["logical_step"], f"{field}.logical_step")
    for name in ("episode_id", "trajectory_id"):
        contract[name] = _optional_identifier(entry[name], f"{field}.{name}")
    contract["recurrent"] = _recurrent_projection(entry["recurrent"], f"{field}.recurrent")
    contract["multi_agent"] = _multi_agent_projection(entry["multi_agent"], f"{field}.multi_agent")

    input_batch = contract["input_schema"]["batch_dimensions"]
    output_batch = contract["output_schema"]["batch_dimensions"]
    if input_batch != output_batch or input_batch != contract["batch_dimensions"]:
        raise ProvenanceSchemaError("interaction_contract batch dimensions must agree across both schemas")
    try:
        _validate_canonical_contract(contract)
    except ProvenanceSchemaError:
        raise
    except (TypeError, ValueError, NotImplementedError) as error:
        raise ProvenanceSchemaError(f"interaction_contract violates canonical invariants: {error}") from error
    return contract


def _validate_canonical_contract(contract: Mapping[str, Any]) -> None:
    """Reapply the authoritative typed contract invariants to decoded evidence."""

    def schema(value: Mapping[str, Any]) -> TensorDictSchema:
        return TensorDictSchema(
            tuple(
                KeySchema(tuple(item["path"]), KeyRole(item["role"]), KeyPresence(item["presence"]))
                for item in value["keys"]
            ),
            BatchSemantics(tuple(value["batch_dimensions"])),
        )

    recurrent_value = contract["recurrent"]
    recurrent = None
    if recurrent_value is not None:
        recurrent = RecurrentSemantics(
            transitions=tuple(
                RecurrentStateTransition(tuple(item["input_key"]), tuple(item["output_key"]))
                for item in recurrent_value["transitions"]
            ),
            reset_keys=tuple(tuple(item) for item in recurrent_value["reset_keys"]),
            sequence_dimension=recurrent_value["sequence_dimension"],
            burn_in=recurrent_value["burn_in"],
            truncated_window=recurrent_value["truncated_window"],
            collector_mode=RecurrentCollectorMode(recurrent_value["collector_mode"]),
        )

    multi_agent_value = contract["multi_agent"]
    multi_agent = None
    if multi_agent_value is not None:
        target = multi_agent_value["target"]
        selector = target["selector"]
        multi_agent = MultiAgentSemantics(
            topology=InteractionTopology(multi_agent_value["topology"]),
            group=multi_agent_value["group"],
            n_agents=multi_agent_value["n_agents"],
            target=SemanticTarget(
                ModelRole(target["role"]),
                AgentSelector(selector["group"], tuple(selector["agents"])),
            ),
        )

    try:
        InteractionContract(
            identity=contract["identity"],
            role=ModelRole(contract["role"]),
            phase=InteractionPhase(contract["phase"]),
            module_path=contract["module_path"],
            input_schema=schema(contract["input_schema"]),
            output_schema=schema(contract["output_schema"]),
            environment=contract["environment"],
            time_dimension=contract["time_dimension"],
            agent_dimension=contract["agent_dimension"],
            objective=contract["objective"],
            exploration_mode=contract["exploration_mode"],
            gradient_enabled=contract["gradient_enabled"],
            inference_mode=contract["inference_mode"],
            autocast_device_type=contract["autocast_device_type"],
            autocast_enabled=contract["autocast_enabled"],
            logical_step=contract["logical_step"],
            episode_id=contract["episode_id"],
            trajectory_id=contract["trajectory_id"],
            model_id=contract["model_id"],
            checkpoint_id=contract["checkpoint_id"],
            module_training=contract["module_training"],
            recurrent=recurrent,
            multi_agent=multi_agent,
        )
    except (TypeError, ValueError, NotImplementedError) as error:
        raise ProvenanceSchemaError(f"interaction_contract violates canonical invariants: {error}") from error


def _schema_projection(value: Any, field: str) -> dict[str, Any]:
    entry = _exact_mapping(value, field, {"keys", "batch_dimensions"})
    keys_value = entry["keys"]
    if not isinstance(keys_value, list):
        raise ProvenanceSchemaError(f"{field}.keys must be an array")
    keys = []
    for index, value in enumerate(keys_value):
        key_field = f"{field}.keys[{index}]"
        key = _exact_mapping(
            value,
            key_field,
            {"path", "role", "presence", "feature_shape", "spec_type", "spec_constraints"},
        )
        feature_shape = key["feature_shape"]
        if feature_shape is not None:
            feature_shape = list(_dimensions(feature_shape, f"{key_field}.feature_shape"))
        constraints = key["spec_constraints"]
        if constraints is not None:
            constraints = _json_object(constraints, f"{key_field}.spec_constraints")
        keys.append(
            {
                "path": list(_key_path_value(key["path"], f"{key_field}.path")),
                "role": _enum_string(key["role"], KeyRole, f"{key_field}.role"),
                "presence": _enum_string(key["presence"], KeyPresence, f"{key_field}.presence"),
                "feature_shape": feature_shape,
                "spec_type": _optional_string(key["spec_type"], f"{key_field}.spec_type"),
                "spec_constraints": constraints,
            }
        )
    return {
        "keys": keys,
        "batch_dimensions": list(_strings(entry["batch_dimensions"], f"{field}.batch_dimensions")),
    }


def _recurrent_projection(value: Any, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    entry = _exact_mapping(
        value,
        field,
        {"transitions", "reset_keys", "sequence_dimension", "burn_in", "truncated_window", "collector_mode"},
    )
    transitions_value = entry["transitions"]
    if not isinstance(transitions_value, list):
        raise ProvenanceSchemaError(f"{field}.transitions must be an array")
    transitions = []
    for index, value in enumerate(transitions_value):
        transition_field = f"{field}.transitions[{index}]"
        transition = _exact_mapping(value, transition_field, {"input_key", "output_key"})
        transitions.append(
            {
                "input_key": list(_key_path_value(transition["input_key"], f"{transition_field}.input_key")),
                "output_key": list(_key_path_value(transition["output_key"], f"{transition_field}.output_key")),
            }
        )
    reset_keys = _key_paths(entry["reset_keys"], f"{field}.reset_keys")
    return {
        "transitions": transitions,
        "reset_keys": [list(path) for path in reset_keys],
        "sequence_dimension": _optional_string(entry["sequence_dimension"], f"{field}.sequence_dimension"),
        "burn_in": _integer(entry["burn_in"], f"{field}.burn_in", minimum=0),
        "truncated_window": _optional_integer(entry["truncated_window"], f"{field}.truncated_window", minimum=1),
        "collector_mode": _enum_string(entry["collector_mode"], RecurrentCollectorMode, f"{field}.collector_mode"),
    }


def _multi_agent_projection(value: Any, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    entry = _exact_mapping(value, field, {"topology", "group", "n_agents", "target"})
    target = _exact_mapping(entry["target"], f"{field}.target", {"role", "selector"})
    selector = _exact_mapping(target["selector"], f"{field}.target.selector", {"group", "agents"})
    agents_value = selector["agents"]
    if not isinstance(agents_value, list) or any(
        type(agent) not in (str, int) or (isinstance(agent, str) and not agent) for agent in agents_value
    ):
        raise ProvenanceSchemaError(f"{field}.target.selector.agents must be an array of identifiers")
    return {
        "topology": _enum_string(entry["topology"], InteractionTopology, f"{field}.topology"),
        "group": _nonempty_string(entry["group"], f"{field}.group"),
        "n_agents": _integer(entry["n_agents"], f"{field}.n_agents", minimum=1),
        "target": {
            "role": _enum_string(target["role"], ModelRole, f"{field}.target.role"),
            "selector": {
                "group": _nonempty_string(selector["group"], f"{field}.target.selector.group"),
                "agents": list(agents_value),
            },
        },
    }


def _validate_lifecycle(events: tuple[LifecycleEvent, ...], contract: Mapping[str, Any], model_passes: int) -> None:
    if len(events) != model_passes * 2:
        raise ProvenanceSchemaError("successful lifecycle evidence must contain one before/after pair per model pass")
    for index, event in enumerate(events):
        if index and event.order != events[index - 1].order + 1:
            raise ProvenanceSchemaError("lifecycle event order must be contiguous and increasing")
        if event.phase.value != contract["phase"] or event.module_path != contract["module_path"]:
            raise ProvenanceSchemaError("lifecycle events must match the interaction phase and module path")
        expected_kind = LifecycleEventType.BEFORE if index % 2 == 0 else LifecycleEventType.AFTER
        if event.kind is not expected_kind or event.error is not None:
            raise ProvenanceSchemaError("successful lifecycle evidence must contain ordered before/after pairs")


def _plan_evidence(value: Any) -> WorkflowPlanEvidence:
    mapping = _exact_mapping(value, "workflow_plan", {"executions", "compatibility"})
    executions_value = mapping["executions"]
    compatibility_value = mapping["compatibility"]
    if not isinstance(executions_value, list) or not isinstance(compatibility_value, list):
        raise ProvenanceSchemaError("workflow_plan executions and compatibility must be arrays")
    executions = []
    for index, item in enumerate(executions_value):
        field = f"workflow_plan.executions[{index}]"
        entry = _exact_mapping(
            item,
            field,
            {"steps", "kind", "in_keys", "out_keys", "model_passes", "gradient_mode", "coexecuted"},
        )
        model_passes = entry["model_passes"]
        if type(model_passes) is not int or model_passes < 0:
            raise ProvenanceSchemaError(f"{field}.model_passes must be a non-negative integer")
        gradient_mode = entry["gradient_mode"]
        if gradient_mode is not None:
            gradient_mode = _nonempty_string(gradient_mode, f"{field}.gradient_mode")
        if type(entry["coexecuted"]) is not bool:
            raise ProvenanceSchemaError(f"{field}.coexecuted must be a boolean")
        executions.append(
            WorkflowExecutionEvidence(
                steps=_strings(entry["steps"], f"{field}.steps"),
                kind=_nonempty_string(entry["kind"], f"{field}.kind"),
                in_keys=_key_paths(entry["in_keys"], f"{field}.in_keys"),
                out_keys=_key_paths(entry["out_keys"], f"{field}.out_keys"),
                model_passes=model_passes,
                gradient_mode=gradient_mode,
                coexecuted=entry["coexecuted"],
            )
        )
    compatibility = []
    for index, item in enumerate(compatibility_value):
        field = f"workflow_plan.compatibility[{index}]"
        entry = _exact_mapping(item, field, {"existing_steps", "candidate_step", "compatible", "reason"})
        if type(entry["compatible"]) is not bool:
            raise ProvenanceSchemaError(f"{field}.compatible must be a boolean")
        compatibility.append(
            WorkflowCompatibilityEvidence(
                existing_steps=_strings(entry["existing_steps"], f"{field}.existing_steps"),
                candidate_step=_nonempty_string(entry["candidate_step"], f"{field}.candidate_step"),
                compatible=entry["compatible"],
                reason=_nonempty_string(entry["reason"], f"{field}.reason"),
            )
        )
    return WorkflowPlanEvidence(tuple(executions), tuple(compatibility))


def _lifecycle_events(value: Any) -> tuple[LifecycleEvent, ...]:
    if not isinstance(value, list):
        raise ProvenanceSchemaError("lifecycle_events must be an array")
    events = []
    for index, item in enumerate(value):
        field = f"lifecycle_events[{index}]"
        entry = _exact_mapping(
            item, field, {"order", "kind", "interaction_id", "phase", "module_path", "key_shapes", "error"}
        )
        order = entry["order"]
        if type(order) is not int or order < 0:
            raise ProvenanceSchemaError(f"{field}.order must be a non-negative integer")
        try:
            kind = LifecycleEventType(entry["kind"])
            phase = InteractionPhase(entry["phase"])
        except (TypeError, ValueError) as error:
            raise ProvenanceSchemaError(f"{field} contains an unknown kind or phase") from error
        shapes = _mapping(entry["key_shapes"], f"{field}.key_shapes")
        parsed_shapes: dict[str, tuple[int, ...] | None] = {}
        for key, shape in shapes.items():
            if shape is None:
                parsed_shapes[key] = None
            elif isinstance(shape, list) and all(type(size) is int and size >= 0 for size in shape):
                parsed_shapes[key] = tuple(shape)
            else:
                raise ProvenanceSchemaError(f"{field}.key_shapes.{key} must be an integer array or null")
        error = entry["error"]
        events.append(
            LifecycleEvent(
                order=order,
                kind=kind,
                interaction_id=_nonempty_string(entry["interaction_id"], f"{field}.interaction_id"),
                phase=phase,
                module_path=_nonempty_string(entry["module_path"], f"{field}.module_path"),
                key_shapes=parsed_shapes,
                error=None if error is None else _nonempty_string(error, f"{field}.error"),
            )
        )
    return tuple(events)


def _required_dependency_names() -> set[str]:
    return {"python", "torch", "tensordict", "torchrl", "tdhook", "xdrl"}


def _validate_artifact_reference(value: Any, field: str) -> None:
    _nonempty_string(value.identity, f"{field}.identity")
    expected_role = InputArtifactRole if isinstance(value, InputArtifactReference) else OutputArtifactRole
    if not isinstance(value.role, expected_role):
        raise ProvenanceSchemaError(f"{field}.role must be a {expected_role.__name__}")
    if not isinstance(value.digest_algorithm, ArtifactDigestAlgorithm):
        raise ProvenanceSchemaError(f"{field}.digest_algorithm must be an ArtifactDigestAlgorithm")
    digest = _nonempty_string(value.digest_value, f"{field}.digest_value")
    expected_length = _DIGEST_HEX_LENGTH[value.digest_algorithm]
    if len(digest) != expected_length or any(character not in "0123456789abcdef" for character in digest):
        raise ProvenanceSchemaError(
            f"{field}.digest_value must be {expected_length} lowercase hexadecimal characters "
            f"for {value.digest_algorithm.value}"
        )
    source = _optional_string(value.source, f"{field}.source")
    revision = _optional_string(value.revision, f"{field}.revision")
    if type(value.source_is_immutable) is not bool:
        raise ProvenanceSchemaError(f"{field}.source_is_immutable must be a boolean")
    if revision is not None and source is None:
        raise ProvenanceSchemaError(f"{field}.revision requires source")
    if value.source_is_immutable and source is None:
        raise ProvenanceSchemaError(f"{field}.source_is_immutable requires source")
    if source is not None and revision is None and not value.source_is_immutable:
        raise ProvenanceSchemaError(f"{field}.source requires a revision unless it is explicitly immutable")
    metadata = _json_object(value.metadata, f"{field}.metadata")
    object.__setattr__(value, "metadata", _freeze_json(metadata))


def _artifact_reference_dict(value: InputArtifactReference | OutputArtifactReference) -> dict[str, Any]:
    return {
        "identity": value.identity,
        "role": value.role.value,
        "digest_algorithm": value.digest_algorithm.value,
        "digest_value": value.digest_value,
        "source": value.source,
        "revision": value.revision,
        "source_is_immutable": value.source_is_immutable,
        "metadata": _thaw_json(value.metadata),
    }


def _artifact_reference_values(
    payload: Mapping[str, Any], field: str, role_type: type[InputArtifactRole] | type[OutputArtifactRole]
) -> dict[str, Any]:
    entry = _exact_mapping(
        payload,
        field,
        {
            "identity",
            "role",
            "digest_algorithm",
            "digest_value",
            "source",
            "revision",
            "source_is_immutable",
            "metadata",
        },
    )
    try:
        role = role_type(entry["role"])
    except (TypeError, ValueError) as error:
        raise ProvenanceSchemaError(f"{field}.role must be a known string value") from error
    try:
        algorithm = ArtifactDigestAlgorithm(entry["digest_algorithm"])
    except (TypeError, ValueError) as error:
        raise ProvenanceSchemaError(f"{field}.digest_algorithm must be a known string value") from error
    return {
        "identity": entry["identity"],
        "role": role,
        "digest_algorithm": algorithm,
        "digest_value": entry["digest_value"],
        "source": entry["source"],
        "revision": entry["revision"],
        "source_is_immutable": entry["source_is_immutable"],
        "metadata": entry["metadata"],
    }


def _decoded_artifacts(value: Any, field: str, artifact_type: Any) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ProvenanceSchemaError(f"{field} must be an array")
    return tuple(artifact_type.from_dict(item) for item in value)


def _artifact_references(
    input_artifacts: Any, output_artifacts: Any
) -> tuple[tuple[InputArtifactReference, ...], tuple[OutputArtifactReference, ...]]:
    if not isinstance(input_artifacts, (list, tuple)) or not all(
        isinstance(item, InputArtifactReference) for item in input_artifacts
    ):
        raise ProvenanceSchemaError("input_artifacts must contain only InputArtifactReference values")
    if not isinstance(output_artifacts, (list, tuple)) or not all(
        isinstance(item, OutputArtifactReference) for item in output_artifacts
    ):
        raise ProvenanceSchemaError("output_artifacts must contain only OutputArtifactReference values")
    identities = [item.identity for item in (*input_artifacts, *output_artifacts)]
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    if duplicates:
        raise ProvenanceSchemaError(f"artifact identities must be unique: {', '.join(duplicates)}")
    return tuple(input_artifacts), tuple(output_artifacts)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ProvenanceSchemaError(f"{field} must be an object with string keys")
    return value


def _exact_mapping(value: Any, field: str, expected: set[str]) -> Mapping[str, Any]:
    mapping = _mapping(value, field)
    missing = expected - set(mapping)
    unknown = set(mapping) - expected
    if missing or unknown:
        raise ProvenanceSchemaError(f"{field} has missing or unknown fields")
    return mapping


def _string_mapping(value: Any, field: str) -> dict[str, str]:
    mapping = _mapping(value, field)
    return {key: _nonempty_string(item, f"{field}.{key}") for key, item in mapping.items()}


def _validate_dependencies(value: Mapping[str, str]) -> dict[str, str]:
    missing = _required_dependency_names() - set(value)
    if missing:
        raise ProvenanceSchemaError(f"dependency provenance is missing: {', '.join(sorted(missing))}")
    dependencies = _string_mapping(value, "dependencies")
    for name, dependency_version in dependencies.items():
        try:
            Version(dependency_version)
        except InvalidVersion as error:
            raise ProvenanceSchemaError(f"dependencies.{name} must be a valid version") from error
    return dependencies


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceSchemaError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, field)


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ProvenanceSchemaError(f"{field} must be a boolean")
    return value


def _optional_boolean(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, field)


def _integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or minimum is not None and value < minimum:
        qualifier = f" greater than or equal to {minimum}" if minimum is not None else ""
        raise ProvenanceSchemaError(f"{field} must be an integer{qualifier}")
    return value


def _optional_integer(value: Any, field: str, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    return _integer(value, field, minimum=minimum)


def _optional_identifier(value: Any, field: str) -> str | int | None:
    if value is None:
        return None
    if type(value) is int:
        return value
    return _nonempty_string(value, field)


def _enum_string(value: Any, enum: type[Enum], field: str) -> str:
    if not isinstance(value, str):
        raise ProvenanceSchemaError(f"{field} must be a known string value")
    try:
        enum(value)
    except ValueError as error:
        raise ProvenanceSchemaError(f"{field} must be a known string value") from error
    return value


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProvenanceSchemaError(f"{field} must be an array")
    return tuple(_nonempty_string(item, f"{field}[{index}]") for index, item in enumerate(value))


def _configured_steps(value: Any) -> tuple[str, ...]:
    """Validate TDHook's ordered, result-affecting step descriptions."""
    if not isinstance(value, list):
        raise ProvenanceSchemaError("configured_steps must be an array")
    return _configured_step_sequence(value)


def _configured_step_sequence(value: Any) -> tuple[str, ...]:
    """Normalize already-constructed configured-step descriptions."""
    if not isinstance(value, (list, tuple)):
        raise ProvenanceSchemaError("configured_steps must be an array")
    return tuple(_nonempty_string(item, f"configured_steps[{index}]") for index, item in enumerate(value))


def _dimensions(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(size) is not int for size in value):
        raise ProvenanceSchemaError(f"{field} must be an array of integers")
    return tuple(value)


def _key_paths(value: Any, field: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        raise ProvenanceSchemaError(f"{field} must be an array")
    paths = []
    for index, path in enumerate(value):
        if not isinstance(path, list) or not path:
            raise ProvenanceSchemaError(f"{field}[{index}] must be a non-empty array")
        paths.append(tuple(_nonempty_string(part, f"{field}[{index}]") for part in path))
    return tuple(paths)


def _key_path_value(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ProvenanceSchemaError(f"{field} must be a non-empty array")
    return tuple(_nonempty_string(part, f"{field}[{index}]") for index, part in enumerate(value))


def _key_path(key: Any) -> tuple[str, ...]:
    return tuple(str(part) for part in key) if isinstance(key, tuple) else (str(key),)


def _json_object(value: Any, field: str) -> dict[str, Any]:
    _mapping(value, field)
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ProvenanceSchemaError(f"{field} must contain only JSON-compatible values") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ProvenanceSchemaError(f"{field} must be an object")
    return decoded


def _json_copy(value: Any, field: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ProvenanceSchemaError(f"{field} must contain only strict JSON-compatible values") from error


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "ArtifactDigestAlgorithm",
    "InputArtifactReference",
    "InputArtifactRole",
    "OutputArtifactReference",
    "OutputArtifactRole",
    "ProvenanceSchemaError",
    "WORKFLOW_PROVENANCE_SCHEMA_REVISION",
    "WorkflowCompatibilityEvidence",
    "WorkflowExecutionEvidence",
    "WorkflowPlanEvidence",
    "WorkflowProvenance",
]
