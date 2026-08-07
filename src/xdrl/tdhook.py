"""Run TDHook v0.2 workflows inside typed TorchRL interactions.

TDHook owns method preparation, hook programs, workflow planning, coexecution,
TensorDict artifacts, and cleanup. XDRL keeps the original model identity and
adds RL-specific execution state, schema validation, lifecycle events, and
plan-to-call-count evidence around every actual root model call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tdhook.execution import GradientMode
from tdhook.workflow import Workflow, WorkflowPlan, WorkflowUpdate

from xdrl.interactions import LifecycleEventType, RuntimeInteractionContext
from xdrl.provenance import WorkflowProvenance


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
    ) -> TDHookWorkflowResult:
        """Execute ``workflow`` and capture versioned plan-to-call provenance."""
        self._validate_boundary(workflow, data)
        validated_dependencies = WorkflowProvenance.validate_run_metadata(
            code_revision=code_revision,
            seed=seed,
            dependencies=dependencies,
        )
        preflight_plan = workflow.plan(self.interaction.module, data)
        if expected_plan is not None and preflight_plan != expected_plan:
            raise RuntimeError("TDHook workflow plan changed after caller preflight")
        self._validate_gradient_contract(preflight_plan)
        self._reject_interaction_module_operators(workflow)
        event_start = len(self.interaction.events)
        with self.interaction, self.interaction.observe_module_calls():
            execute = getattr(workflow, "run_with_evidence", workflow.run)
            execution = execute(self.interaction.module, data)
        if isinstance(execution, TensorDictBase):
            # TDHook 0.2.0 compatibility.  This uses only public surfaces and
            # fails closed if a second public plan no longer agrees.
            result = execution
            plan = workflow.plan(self.interaction.module, data)
            configured_steps = tuple(
                f"{index}:{type(item.step if isinstance(item, WorkflowUpdate) else item).__name__}"
                for index, item in enumerate(workflow.steps)
            )
            if plan != preflight_plan:
                raise RuntimeError("TDHook workflow plan changed during execution")
        else:
            result = execution.data
            plan = execution.plan
            configured_steps = execution.configured_steps
            if not isinstance(result, TensorDictBase) or not isinstance(plan, WorkflowPlan):
                raise TypeError("TDHook workflow execution returned invalid public evidence")
        if expected_plan is not None and plan != expected_plan:
            raise RuntimeError("TDHook workflow plan changed during execution")
        events = tuple(self.interaction.events[event_start:])
        model_calls = sum(event.kind is LifecycleEventType.AFTER for event in events)
        if model_calls != plan.model_passes:
            raise RuntimeError(
                f"TDHook workflow model-pass mismatch: plan declares {plan.model_passes}, XDRL observed {model_calls}"
            )
        provenance = WorkflowProvenance.capture(
            self.interaction.contract,
            plan,
            configured_steps,
            events,
            code_revision=code_revision,
            seed=seed,
            dependencies=validated_dependencies,
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
                raise ValueError(
                    "gradient-required TDHook workflows are unsupported because Workflow.run removes hook bindings "
                    "before a caller-managed backward pass"
                )
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


def _reject_unsupported_module(module: torch.nn.Module) -> None:
    if any(hasattr(child, "_orig_mod") for child in module.modules()):
        raise NotImplementedError("torch.compile modules are not supported by the TDHook workflow runner")
    distributed_types = (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)
    if any(isinstance(child, distributed_types) for child in module.modules()):
        raise NotImplementedError("distributed modules are not supported by the TDHook workflow runner")
    if type(module).__module__.startswith("torch.distributed.rpc"):
        raise NotImplementedError("remote modules are not supported by the TDHook workflow runner")


__all__ = ["TDHookWorkflowResult", "TDHookWorkflowRunner"]
