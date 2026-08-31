"""Run TDHook v0.2 workflows inside typed TorchRL interactions.

TDHook owns method preparation, hook programs, workflow planning, coexecution,
TensorDict artifacts, and cleanup. XDRL keeps the original model identity and
adds RL-specific execution state, schema validation, lifecycle events, and
plan-to-call-count evidence around every actual root model call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tdhook.execution import GradientMode
from tdhook.workflow import Workflow, WorkflowPlan, WorkflowUpdate

from xdrl.interactions import LifecycleEventType, RuntimeInteractionContext
from xdrl.provenance import (
    InputArtifactReference,
    OutputArtifactDeclaration,
    OutputArtifactDigest,
    WorkflowProvenance,
)

OutputArtifactResolver = Callable[
    [TensorDictBase, tuple[OutputArtifactDeclaration, ...]],
    tuple[OutputArtifactDigest, ...],
]


@dataclass(frozen=True, slots=True)
class TDHookWorkflowResult:
    """Native TDHook output, exact plan, and versioned XDRL provenance."""

    data: TensorDictBase
    plan: WorkflowPlan
    provenance: WorkflowProvenance


@dataclass(frozen=True, slots=True)
class WorkflowStepDifference:
    """One declared configured-step difference between paired workflow arms."""

    index: int
    baseline: str | None
    intervention: str | None

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("workflow step difference index must be a non-negative integer")
        if self.baseline == self.intervention:
            raise ValueError("workflow step difference must contain distinct arm descriptions")


@dataclass(frozen=True, slots=True)
class WorkflowArmReference:
    """Tensor-free references to one paired arm's durable evidence."""

    provenance_sha256: str
    input_artifacts: tuple[str, ...]
    output_artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.provenance_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.provenance_sha256
        ):
            raise ValueError("paired arm provenance_sha256 must be a lowercase SHA-256 digest")
        for values, field in (
            (self.input_artifacts, "input_artifacts"),
            (self.output_artifacts, "output_artifacts"),
        ):
            if not isinstance(values, tuple) or any(not isinstance(value, str) or not value for value in values):
                raise TypeError(f"paired arm {field} must be a tuple of non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"paired arm {field} identities must be unique")


@dataclass(frozen=True, slots=True)
class TDHookWorkflowPairManifest:
    """Tensor-free evidence that two TDHook workflow arms were matched.

    The manifest records execution mechanics and provenance references only. It
    deliberately makes no claim that a difference between outputs is causal.
    """

    schema_revision: int
    pair_id: str
    interaction_id: str
    model_id: str
    checkpoint_id: str
    changed_steps: tuple[WorkflowStepDifference, ...]
    baseline: WorkflowArmReference
    intervention: WorkflowArmReference
    matched_rng_sources: tuple[str, ...]
    unsupported_randomness_sources: tuple[str, ...]
    interpretation: str = "mechanics_and_provenance_only"

    def __post_init__(self) -> None:
        if self.schema_revision != 1:
            raise ValueError("unsupported paired workflow manifest schema revision")
        for value, field in (
            (self.pair_id, "pair_id"),
            (self.interaction_id, "interaction_id"),
            (self.model_id, "model_id"),
            (self.checkpoint_id, "checkpoint_id"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
        if not all(isinstance(item, WorkflowStepDifference) for item in self.changed_steps):
            raise TypeError("changed_steps must contain WorkflowStepDifference values")
        if tuple(item.index for item in self.changed_steps) != tuple(
            sorted({item.index for item in self.changed_steps})
        ):
            raise ValueError("paired workflow changed step indices must be unique and sorted")
        if not isinstance(self.baseline, WorkflowArmReference) or not isinstance(
            self.intervention, WorkflowArmReference
        ):
            raise TypeError("paired workflow arms must be WorkflowArmReference values")
        for sources, field in (
            (self.matched_rng_sources, "matched_rng_sources"),
            (self.unsupported_randomness_sources, "unsupported_randomness_sources"),
        ):
            if not isinstance(sources, tuple) or any(not isinstance(source, str) or not source for source in sources):
                raise TypeError(f"{field} must be a tuple of non-empty strings")
            if len(sources) != len(set(sources)):
                raise ValueError(f"{field} must contain unique values")
        if self.interpretation != "mechanics_and_provenance_only":
            raise ValueError("paired workflow manifests cannot assign a causal interpretation")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Encode the pair manifest deterministically."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TDHookWorkflowPairManifest:
        """Decode one versioned pair manifest without reconstructing tensors."""
        if not isinstance(payload, Mapping):
            raise TypeError("paired workflow manifest must be an object")
        expected = {
            "schema_revision",
            "pair_id",
            "interaction_id",
            "model_id",
            "checkpoint_id",
            "changed_steps",
            "baseline",
            "intervention",
            "matched_rng_sources",
            "unsupported_randomness_sources",
            "interpretation",
        }
        if set(payload) != expected:
            raise ValueError("paired workflow manifest has missing or unknown fields")
        changed_steps = payload["changed_steps"]
        if not isinstance(changed_steps, (list, tuple)):
            raise TypeError("changed_steps must be an array")
        arms = []
        for name in ("baseline", "intervention"):
            arm = payload[name]
            if not isinstance(arm, Mapping) or set(arm) != {
                "provenance_sha256",
                "input_artifacts",
                "output_artifacts",
            }:
                raise ValueError(f"{name} arm reference has missing or unknown fields")
            arms.append(
                WorkflowArmReference(
                    provenance_sha256=arm["provenance_sha256"],
                    input_artifacts=tuple(arm["input_artifacts"]),
                    output_artifacts=tuple(arm["output_artifacts"]),
                )
            )
        return cls(
            schema_revision=payload["schema_revision"],
            pair_id=payload["pair_id"],
            interaction_id=payload["interaction_id"],
            model_id=payload["model_id"],
            checkpoint_id=payload["checkpoint_id"],
            changed_steps=tuple(WorkflowStepDifference(**item) for item in changed_steps),
            baseline=arms[0],
            intervention=arms[1],
            matched_rng_sources=tuple(payload["matched_rng_sources"]),
            unsupported_randomness_sources=tuple(payload["unsupported_randomness_sources"]),
            interpretation=payload["interpretation"],
        )

    @classmethod
    def from_json(cls, payload: str) -> TDHookWorkflowPairManifest:
        """Decode one JSON pair manifest."""
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise TypeError("paired workflow manifest JSON must contain an object")
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class TDHookPairedWorkflowResult:
    """Native results for both arms plus their tensor-free pair manifest."""

    baseline: TDHookWorkflowResult
    intervention: TDHookWorkflowResult
    manifest: TDHookWorkflowPairManifest


class PairedWorkflowValidationError(ValueError):
    """Two workflow arms cannot be compared under the declared pair contract."""


class PairedWorkflowExecutionError(RuntimeError):
    """One paired arm failed after pair validation and state preflight."""

    def __init__(
        self,
        arm: str,
        *,
        baseline: TDHookWorkflowResult | None = None,
    ) -> None:
        super().__init__(f"paired TDHook workflow {arm} arm failed")
        self.arm = arm
        self.baseline = baseline


@dataclass(slots=True)
class TDHookWorkflowRunner:
    """Delegate TDHook v0.2 workflow planning and execution through XDRL."""

    interaction: RuntimeInteractionContext

    def __post_init__(self) -> None:
        _reject_unsupported_module(self.interaction.module)

    def materialize(self, tensordict: TensorDictBase | None = None) -> None:
        """Explicitly materialise lazy parameters before planning or execution."""
        if not _has_uninitialized_parameters(self.interaction.module):
            return
        data = self.interaction.representative_input if tensordict is None else tensordict
        self.interaction.input_schema.validate_inputs(data)
        self.interaction.module(data.clone())
        if _has_uninitialized_parameters(self.interaction.module):
            raise RuntimeError("interaction module remains lazy after materialisation")

    def plan(self, workflow: Workflow, data: TensorDictBase) -> WorkflowPlan:
        """Return TDHook's immutable plan after XDRL input preflight."""
        self._validate_boundary(workflow, data)
        return workflow.plan(self.interaction.module, data)

    def run(
        self,
        workflow: Workflow,
        data: TensorDictBase,
        *,
        code_revision: str,
        expected_plan: WorkflowPlan | None = None,
        seed: int | None = None,
        dependencies: Mapping[str, str] | None = None,
        input_artifacts: tuple[InputArtifactReference, ...] = (),
        output_artifacts: tuple[OutputArtifactDeclaration, ...] = (),
        output_artifact_resolver: OutputArtifactResolver | None = None,
        callback_identifiers: Mapping[Callable[..., object], str] | None = None,
    ) -> TDHookWorkflowResult:
        """Execute ``workflow`` and capture versioned plan-to-call provenance.

        Output identities and roles are declared before execution. Their resolver
        runs only after successful TDHook execution and returns the digests that
        are bound to those exact declarations in provenance.
        """
        self._validate_boundary(workflow, data)
        validated_dependencies = WorkflowProvenance.validate_run_metadata(
            code_revision=code_revision,
            seed=seed,
            dependencies=dependencies,
            input_artifacts=input_artifacts,
        )
        output_artifacts = WorkflowProvenance.validate_output_artifact_declarations(input_artifacts, output_artifacts)
        if bool(output_artifacts) != (output_artifact_resolver is not None):
            raise ValueError("output artifact declarations and output_artifact_resolver must be provided together")
        if output_artifact_resolver is not None and not callable(output_artifact_resolver):
            raise TypeError("output_artifact_resolver must be callable")
        preflight_plan = workflow.plan(self.interaction.module, data)
        if expected_plan is not None and preflight_plan != expected_plan:
            raise RuntimeError("TDHook workflow plan changed after caller preflight")
        self._validate_gradient_contract(preflight_plan)
        self._reject_interaction_module_operators(workflow)
        try:
            describe = workflow.describe
            run_with_plan = workflow.run_with_plan
        except AttributeError as error:
            raise RuntimeError(
                "TDHookWorkflowRunner requires TDHook public execution evidence; install the supported TDHook revision"
            ) from error
        if not callable(describe) or not callable(run_with_plan):
            raise RuntimeError(
                "TDHookWorkflowRunner requires TDHook public execution evidence; install the supported TDHook revision"
            )
        configured_steps = tuple(
            _configured_step_description(description)
            for description in describe(
                self.interaction.module,
                data,
                callback_identifiers=callback_identifiers,
            )
        )
        event_start = len(self.interaction.events)
        with self.interaction, self.interaction.observe_module_calls():
            execution = run_with_plan(self.interaction.module, data)
        result = execution.data
        plan = execution.plan
        if not isinstance(result, TensorDictBase) or not isinstance(plan, WorkflowPlan):
            raise TypeError("TDHook workflow execution returned invalid public evidence")
        if plan != preflight_plan:
            raise RuntimeError("TDHook workflow plan changed during execution")
        events = tuple(self.interaction.events[event_start:])
        model_calls = sum(event.kind is LifecycleEventType.AFTER for event in events)
        if model_calls != plan.model_passes:
            raise RuntimeError(
                f"TDHook workflow model-pass mismatch: plan declares {plan.model_passes}, XDRL observed {model_calls}"
            )
        resolved_output_artifacts = ()
        if output_artifact_resolver is not None:
            resolved_output_artifacts = WorkflowProvenance.resolve_output_artifacts(
                output_artifacts,
                output_artifact_resolver(result, output_artifacts),
            )
        provenance = WorkflowProvenance.capture(
            self.interaction.contract,
            plan,
            configured_steps,
            events,
            code_revision=code_revision,
            seed=seed,
            dependencies=validated_dependencies,
            input_artifacts=input_artifacts,
            output_artifacts=resolved_output_artifacts,
        )
        return TDHookWorkflowResult(result, plan, provenance)

    def run_paired(
        self,
        baseline_workflow: Workflow,
        intervention_workflow: Workflow,
        data: TensorDictBase,
        *,
        pair_id: str,
        code_revision: str,
        declared_workflow_differences: tuple[int, ...] = (),
        intervention_runner: TDHookWorkflowRunner | None = None,
        expected_baseline_plan: WorkflowPlan | None = None,
        expected_intervention_plan: WorkflowPlan | None = None,
        seed: int | None = None,
        dependencies: Mapping[str, str] | None = None,
        input_artifacts: tuple[InputArtifactReference, ...] = (),
        baseline_output_artifacts: tuple[OutputArtifactDeclaration, ...] = (),
        baseline_output_artifact_resolver: OutputArtifactResolver | None = None,
        intervention_output_artifacts: tuple[OutputArtifactDeclaration, ...] = (),
        intervention_output_artifact_resolver: OutputArtifactResolver | None = None,
        callback_identifiers: Mapping[Callable[..., object], str] | None = None,
    ) -> TDHookPairedWorkflowResult:
        """Run matched baseline and intervention workflows without state leakage.

        Only the process-global PyTorch CPU generator and, when available, all
        process-global CUDA generators are matched. Python, NumPy, application
        generators, and other external randomness sources remain caller-owned
        and are declared as unsupported in the returned manifest.
        """
        other = self if intervention_runner is None else intervention_runner
        _validate_pair_contract(self, other, pair_id)
        self._validate_boundary(baseline_workflow, data)
        other._validate_boundary(intervention_workflow, data)
        _validate_paired_operators(baseline_workflow, intervention_workflow)
        baseline_steps = _workflow_descriptions(
            baseline_workflow, self.interaction.module, data, callback_identifiers=callback_identifiers
        )
        intervention_steps = _workflow_descriptions(
            intervention_workflow, other.interaction.module, data, callback_identifiers=callback_identifiers
        )
        differences = _workflow_differences(baseline_steps, intervention_steps)
        declared = _declared_difference_indices(declared_workflow_differences)
        actual = tuple(item.index for item in differences)
        if actual != declared:
            raise PairedWorkflowValidationError(
                f"workflow differences do not match declaration: declared {declared!r}, observed {actual!r}"
            )

        initial_rng = _rng_state()
        final_rng = initial_rng
        module_state = _ModuleStateSnapshot.capture(self.interaction.module)
        baseline: TDHookWorkflowResult | None = None
        arm = "baseline"
        try:
            baseline = self.run(
                baseline_workflow,
                data.clone(),
                code_revision=code_revision,
                expected_plan=expected_baseline_plan,
                seed=seed,
                dependencies=dependencies,
                input_artifacts=input_artifacts,
                output_artifacts=baseline_output_artifacts,
                output_artifact_resolver=baseline_output_artifact_resolver,
                callback_identifiers=callback_identifiers,
            )
            final_rng = _rng_state()
            module_state.restore(self.interaction.module)
            _set_rng_state(initial_rng)
            arm = "intervention"
            intervention = other.run(
                intervention_workflow,
                data.clone(),
                code_revision=code_revision,
                expected_plan=expected_intervention_plan,
                seed=seed,
                dependencies=dependencies,
                input_artifacts=input_artifacts,
                output_artifacts=intervention_output_artifacts,
                output_artifact_resolver=intervention_output_artifact_resolver,
                callback_identifiers=callback_identifiers,
            )
        except Exception as error:
            raise PairedWorkflowExecutionError(arm, baseline=baseline) from error
        finally:
            module_state.restore(self.interaction.module)
            _set_rng_state(final_rng)

        manifest = TDHookWorkflowPairManifest(
            schema_revision=1,
            pair_id=pair_id,
            interaction_id=self.interaction.contract.identity,
            model_id=self.interaction.contract.model_id or "",
            checkpoint_id=self.interaction.contract.checkpoint_id or "",
            changed_steps=differences,
            baseline=_arm_reference(baseline),
            intervention=_arm_reference(intervention),
            matched_rng_sources=("torch_cpu",) + (("torch_cuda",) if initial_rng[1] is not None else ()),
            unsupported_randomness_sources=(
                "python_random",
                "numpy_random",
                "explicit_torch_generators",
                "external_or_application_randomness",
            ),
        )
        return TDHookPairedWorkflowResult(baseline, intervention, manifest)

    def _validate_boundary(self, workflow: Workflow, data: TensorDictBase) -> None:
        if not isinstance(workflow, Workflow):
            raise TypeError(f"workflow must be a TDHook Workflow, got {type(workflow).__name__}")
        if not isinstance(data, TensorDictBase):
            raise TypeError(f"workflow data must be a TensorDict, got {type(data).__name__}")
        if _has_uninitialized_parameters(self.interaction.module):
            raise RuntimeError("interaction module has lazy parameters; call materialize() before workflow use")
        self.interaction.input_schema.validate_inputs(data)

    def _validate_gradient_contract(self, plan: WorkflowPlan) -> None:
        contract = self.interaction.contract
        for execution in plan.executions:
            if execution.gradient_mode is GradientMode.REQUIRED:
                autograd_lifetime = getattr(execution, "autograd_lifetime", None)
                if getattr(autograd_lifetime, "value", autograd_lifetime) == "backward":
                    raise ValueError(
                        "deferred-backward TDHook workflows are unsupported because XDRL does not own the "
                        "caller-managed backward lifecycle"
                    )
                if autograd_lifetime is None:
                    raise ValueError(
                        "gradient-required TDHook execution requires a TDHook autograd lifetime declaration"
                    )
                if contract.inference_mode:
                    raise ValueError("gradient-required TDHook execution is incompatible with inference_mode=True")
                if not contract.gradient_enabled:
                    raise ValueError("gradient-required TDHook execution requires gradient_enabled=True")
            if execution.gradient_mode is GradientMode.DISABLED and contract.gradient_enabled:
                raise ValueError("gradient-disabled TDHook execution is incompatible with gradient_enabled=True")

    def _reject_interaction_module_operators(self, workflow: Workflow) -> None:
        for wrapped_step in workflow.steps:
            step = wrapped_step.step if isinstance(wrapped_step, WorkflowUpdate) else wrapped_step
            if isinstance(step, TensorDictModuleBase) and any(
                child is self.interaction.module for child in step.modules()
            ):
                raise ValueError(
                    "workflow operators must not invoke the interaction module; use a TDHook method for root calls"
                )


def _has_uninitialized_parameters(module: torch.nn.Module) -> bool:
    uninitialized = (torch.nn.parameter.UninitializedParameter, torch.nn.parameter.UninitializedBuffer)
    return any(isinstance(value, uninitialized) for value in (*module.parameters(), *module.buffers()))


def _configured_step_description(description: object) -> str:
    """Serialize one TDHook public configured-step description deterministically."""
    to_dict = getattr(description, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("TDHook configured-step descriptions must expose to_dict()")
    import json

    try:
        return json.dumps(to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError("TDHook configured-step description must be JSON-compatible") from error


def _workflow_descriptions(
    workflow: Workflow,
    module: torch.nn.Module,
    data: TensorDictBase,
    *,
    callback_identifiers: Mapping[Callable[..., object], str] | None,
) -> tuple[str, ...]:
    try:
        descriptions = workflow.describe(module, data, callback_identifiers=callback_identifiers)
    except AttributeError as error:
        raise RuntimeError(
            "TDHookWorkflowRunner requires TDHook public execution evidence; install the supported TDHook revision"
        ) from error
    configured = iter(descriptions)
    results = []
    for wrapped_step in workflow.steps:
        step = wrapped_step.step if isinstance(wrapped_step, WorkflowUpdate) else wrapped_step
        if isinstance(step, TensorDictModuleBase):
            continue
        description = _configured_step_description(next(configured))
        results.append(
            json.dumps(
                {
                    "allow_output_overwrite": isinstance(wrapped_step, WorkflowUpdate),
                    "description": json.loads(description),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    try:
        next(configured)
    except StopIteration:
        return tuple(results)
    raise RuntimeError("TDHook workflow descriptions do not match configured method steps")


def _validate_paired_operators(baseline: Workflow, intervention: Workflow) -> None:
    """Require non-method operators to be literally shared between both arms."""
    size = max(len(baseline.steps), len(intervention.steps))
    for index in range(size):
        left = baseline.steps[index] if index < len(baseline.steps) else None
        right = intervention.steps[index] if index < len(intervention.steps) else None
        left_step = left.step if isinstance(left, WorkflowUpdate) else left
        right_step = right.step if isinstance(right, WorkflowUpdate) else right
        if isinstance(left_step, TensorDictModuleBase) or isinstance(right_step, TensorDictModuleBase):
            if left_step is not right_step or isinstance(left, WorkflowUpdate) is not isinstance(
                right, WorkflowUpdate
            ):
                raise PairedWorkflowValidationError(
                    "paired workflow TensorDict operators must be the same instances at the same indices"
                )


def _workflow_differences(
    baseline: tuple[str, ...], intervention: tuple[str, ...]
) -> tuple[WorkflowStepDifference, ...]:
    return tuple(
        WorkflowStepDifference(
            index,
            baseline[index] if index < len(baseline) else None,
            intervention[index] if index < len(intervention) else None,
        )
        for index in range(max(len(baseline), len(intervention)))
        if (baseline[index] if index < len(baseline) else None)
        != (intervention[index] if index < len(intervention) else None)
    )


def _declared_difference_indices(indices: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(indices, tuple) or any(type(index) is not int or index < 0 for index in indices):
        raise PairedWorkflowValidationError("declared_workflow_differences must be a tuple of non-negative integers")
    if indices != tuple(sorted(set(indices))):
        raise PairedWorkflowValidationError("declared workflow difference indices must be unique and sorted")
    return indices


def _validate_pair_contract(
    baseline: TDHookWorkflowRunner,
    intervention: TDHookWorkflowRunner,
    pair_id: str,
) -> None:
    if not isinstance(pair_id, str) or not pair_id:
        raise PairedWorkflowValidationError("pair_id must be a non-empty string")
    left = baseline.interaction.contract
    right = intervention.interaction.contract
    if left.model_id is None or left.checkpoint_id is None:
        raise PairedWorkflowValidationError("paired workflows require explicit model and checkpoint identities")
    if left.identity != right.identity or left.model_id != right.model_id or left.checkpoint_id != right.checkpoint_id:
        raise PairedWorkflowValidationError(
            "paired workflows must share interaction, model, and checkpoint identities"
        )
    if left.to_dict() != right.to_dict():
        raise PairedWorkflowValidationError("paired workflows require identical interaction contracts")
    if baseline.interaction.module is not intervention.interaction.module:
        raise PairedWorkflowValidationError("paired workflows must execute against the same module instance")


@dataclass(frozen=True, slots=True)
class _ModuleStateSnapshot:
    state: Mapping[str, torch.Tensor]
    training: tuple[bool, ...]
    gradients: tuple[torch.Tensor | None, ...]

    @classmethod
    def capture(cls, module: torch.nn.Module) -> _ModuleStateSnapshot:
        return cls(
            {name: value.detach().clone() for name, value in module.state_dict().items()},
            tuple(child.training for child in module.modules()),
            tuple(
                parameter.grad.detach().clone() if parameter.grad is not None else None
                for parameter in module.parameters()
            ),
        )

    def restore(self, module: torch.nn.Module) -> None:
        module.load_state_dict(self.state, strict=True)
        children = tuple(module.modules())
        if len(children) != len(self.training):
            raise RuntimeError("paired workflow changed the module hierarchy")
        for child, training in zip(children, self.training, strict=True):
            child.training = training
        parameters = tuple(module.parameters())
        if len(parameters) != len(self.gradients):
            raise RuntimeError("paired workflow changed the module parameter structure")
        for parameter, gradient in zip(parameters, self.gradients, strict=True):
            parameter.grad = gradient.detach().clone() if gradient is not None else None


def _rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    return torch.get_rng_state().clone(), (
        [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None
    )


def _set_rng_state(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def _arm_reference(result: TDHookWorkflowResult) -> WorkflowArmReference:
    provenance = result.provenance
    return WorkflowArmReference(
        provenance_sha256=hashlib.sha256(provenance.to_json().encode()).hexdigest(),
        input_artifacts=tuple(item.identity for item in provenance.input_artifacts),
        output_artifacts=tuple(item.identity for item in provenance.output_artifacts),
    )


def _reject_unsupported_module(module: torch.nn.Module) -> None:
    if any(hasattr(child, "_orig_mod") for child in module.modules()):
        raise NotImplementedError("torch.compile modules are not supported by the TDHook workflow runner")
    distributed_types = (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)
    if any(isinstance(child, distributed_types) for child in module.modules()):
        raise NotImplementedError("distributed modules are not supported by the TDHook workflow runner")
    if type(module).__module__.startswith("torch.distributed.rpc"):
        raise NotImplementedError("remote modules are not supported by the TDHook workflow runner")


__all__ = [
    "OutputArtifactResolver",
    "PairedWorkflowExecutionError",
    "PairedWorkflowValidationError",
    "TDHookPairedWorkflowResult",
    "TDHookWorkflowPairManifest",
    "TDHookWorkflowResult",
    "TDHookWorkflowRunner",
    "WorkflowArmReference",
    "WorkflowStepDifference",
]
