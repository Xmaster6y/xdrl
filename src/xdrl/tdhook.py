"""Run TDHook v0.2 workflows inside typed TorchRL interactions.

TDHook owns method preparation, hook programs, workflow planning, coexecution,
TensorDict artifacts, and cleanup. XDRL keeps the original model identity and
adds RL-specific execution state, schema validation, lifecycle events, and
plan-to-call-count evidence around every actual root model call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

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
            _configured_step_description(description) for description in describe(self.interaction.module, data)
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


def _reject_unsupported_module(module: torch.nn.Module) -> None:
    if any(hasattr(child, "_orig_mod") for child in module.modules()):
        raise NotImplementedError("torch.compile modules are not supported by the TDHook workflow runner")
    distributed_types = (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)
    if any(isinstance(child, distributed_types) for child in module.modules()):
        raise NotImplementedError("distributed modules are not supported by the TDHook workflow runner")
    if type(module).__module__.startswith("torch.distributed.rpc"):
        raise NotImplementedError("remote modules are not supported by the TDHook workflow runner")


__all__ = ["OutputArtifactResolver", "TDHookWorkflowResult", "TDHookWorkflowRunner"]
