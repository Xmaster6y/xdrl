"""Versioned, serialisable evidence for TDHook workflows in XDRL."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from packaging.version import InvalidVersion, Version
from tdhook.workflow import CompatibilityDecision, PlannedExecution, WorkflowPlan

from xdrl.compatibility import installed_dependency_versions
from xdrl.interactions import (
    InteractionContract,
    InteractionPhase,
    LifecycleEvent,
    LifecycleEventType,
)

WORKFLOW_PROVENANCE_SCHEMA_REVISION = 1


class ProvenanceSchemaError(ValueError):
    """A workflow provenance payload violates the current schema."""


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
    lifecycle_events: tuple[LifecycleEvent, ...]
    dependencies: Mapping[str, str]
    code_revision: str
    seed: int | None = None

    def __post_init__(self) -> None:
        if type(self.schema_revision) is not int or self.schema_revision != WORKFLOW_PROVENANCE_SCHEMA_REVISION:
            raise ProvenanceSchemaError(
                f"unsupported workflow provenance schema revision {self.schema_revision}; "
                f"expected {WORKFLOW_PROVENANCE_SCHEMA_REVISION}"
            )
        if not isinstance(self.interaction_contract, Mapping):
            raise ProvenanceSchemaError("interaction_contract must be an object")
        interaction_id = _nonempty_string(self.interaction_contract.get("identity"), "interaction_contract.identity")
        _nonempty_string(self.code_revision, "code_revision")
        if self.seed is not None and type(self.seed) is not int:
            raise ProvenanceSchemaError("seed must be an integer or null")
        _validate_dependencies(self.dependencies)
        if any(event.interaction_id != interaction_id for event in self.lifecycle_events):
            raise ProvenanceSchemaError("lifecycle events must belong to the interaction contract")
        if self.model_calls != self.workflow_plan.model_passes:
            raise ProvenanceSchemaError(
                "workflow plan and lifecycle evidence disagree: "
                f"plan declares {self.workflow_plan.model_passes}, events contain {self.model_calls} successful calls"
            )
        try:
            json.dumps(self.to_dict(), sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ProvenanceSchemaError(f"workflow provenance contains a non-serialisable value: {error}") from error

    @classmethod
    def validate_run_metadata(
        cls,
        *,
        code_revision: str,
        seed: int | None = None,
        dependencies: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Validate reproduction metadata before a workflow mutates live state."""
        _nonempty_string(code_revision, "code_revision")
        if seed is not None and type(seed) is not int:
            raise ProvenanceSchemaError("seed must be an integer or null")
        resolved = installed_dependency_versions() if dependencies is None else dependencies
        return _validate_dependencies(dict(resolved))

    @classmethod
    def capture(
        cls,
        contract: InteractionContract,
        plan: WorkflowPlan,
        events: tuple[LifecycleEvent, ...],
        *,
        code_revision: str,
        seed: int | None = None,
        dependencies: Mapping[str, str] | None = None,
    ) -> WorkflowProvenance:
        """Capture verified workflow evidence from public XDRL and TDHook state."""
        validated_dependencies = cls.validate_run_metadata(
            code_revision=code_revision,
            seed=seed,
            dependencies=dependencies,
        )
        return cls(
            schema_revision=WORKFLOW_PROVENANCE_SCHEMA_REVISION,
            interaction_contract=json.loads(json.dumps(contract.to_dict())),
            workflow_plan=WorkflowPlanEvidence.from_plan(plan),
            lifecycle_events=events,
            dependencies=validated_dependencies,
            code_revision=code_revision,
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
            "interaction_contract": json.loads(json.dumps(self.interaction_contract)),
            "workflow_plan": asdict(self.workflow_plan),
            "lifecycle_events": [event.to_dict() for event in self.lifecycle_events],
            "dependencies": dict(self.dependencies),
            "code_revision": self.code_revision,
            "seed": self.seed,
        }
        return json.loads(json.dumps(payload))

    def to_json(self) -> str:
        """Encode workflow provenance deterministically."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkflowProvenance:
        """Decode provenance while rejecting unknown revisions or fields."""
        if not isinstance(payload, Mapping):
            raise ProvenanceSchemaError("workflow provenance must contain an object")
        required = {item.name for item in fields(cls)}
        missing = required - set(payload)
        unknown = set(payload) - required
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing fields: {', '.join(sorted(missing))}")
            if unknown:
                details.append(f"unknown fields: {', '.join(sorted(unknown))}")
            raise ProvenanceSchemaError("; ".join(details))
        revision = payload["schema_revision"]
        if type(revision) is not int or revision != WORKFLOW_PROVENANCE_SCHEMA_REVISION:
            raise ProvenanceSchemaError(
                f"unsupported workflow provenance schema revision {revision!r}; "
                f"expected {WORKFLOW_PROVENANCE_SCHEMA_REVISION}"
            )
        contract = dict(_mapping(payload["interaction_contract"], "interaction_contract"))
        plan = _plan_evidence(payload["workflow_plan"])
        events = _lifecycle_events(payload["lifecycle_events"])
        dependencies = _validate_dependencies(_string_mapping(payload["dependencies"], "dependencies"))
        seed = payload["seed"]
        if seed is not None and type(seed) is not int:
            raise ProvenanceSchemaError("seed must be an integer or null")
        return cls(
            schema_revision=revision,
            interaction_contract=contract,
            workflow_plan=plan,
            lifecycle_events=events,
            dependencies=dependencies,
            code_revision=_nonempty_string(payload["code_revision"], "code_revision"),
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


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProvenanceSchemaError(f"{field} must be an array")
    return tuple(_nonempty_string(item, f"{field}[{index}]") for index, item in enumerate(value))


def _key_paths(value: Any, field: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        raise ProvenanceSchemaError(f"{field} must be an array")
    paths = []
    for index, path in enumerate(value):
        if not isinstance(path, list) or not path:
            raise ProvenanceSchemaError(f"{field}[{index}] must be a non-empty array")
        paths.append(tuple(_nonempty_string(part, f"{field}[{index}]") for part in path))
    return tuple(paths)


def _key_path(key: Any) -> tuple[str, ...]:
    return tuple(str(part) for part in key) if isinstance(key, tuple) else (str(key),)


__all__ = [
    "ProvenanceSchemaError",
    "WORKFLOW_PROVENANCE_SCHEMA_REVISION",
    "WorkflowCompatibilityEvidence",
    "WorkflowExecutionEvidence",
    "WorkflowPlanEvidence",
    "WorkflowProvenance",
]
