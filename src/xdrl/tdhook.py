"""Run TDHook v0.2 workflows inside typed TorchRL interactions.

TDHook owns method preparation, hook programs, workflow planning, coexecution,
TensorDict artifacts, and cleanup. XDRL keeps the original model identity and
adds RL-specific execution state, schema validation, lifecycle events, and
plan-to-call-count evidence around every actual root model call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from tensordict import TensorDictBase
from tdhook.execution import GradientMode
from tdhook.workflow import Workflow, WorkflowPlan

from xdrl.interactions import LifecycleEvent, LifecycleEventType, RuntimeInteractionContext


@dataclass(frozen=True, slots=True)
class WorkflowRunRecord:
    """Tensor-free evidence associating one TDHook plan with an RL interaction."""

    interaction_id: str
    plan: WorkflowPlan
    events: tuple[LifecycleEvent, ...]

    @property
    def model_calls(self) -> int:
        """Return the number of successful root model calls observed by XDRL."""
        return sum(event.kind is LifecycleEventType.AFTER for event in self.events)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible projection of the public plan and events."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TDHookWorkflowResult:
    """The native TDHook TensorDict result paired with XDRL execution evidence."""

    data: TensorDictBase
    record: WorkflowRunRecord

    @property
    def plan(self) -> WorkflowPlan:
        """Return the exact public TDHook plan associated with this execution."""
        return self.record.plan


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
        expected_plan: WorkflowPlan | None = None,
    ) -> TDHookWorkflowResult:
        """Execute ``workflow`` while validating every actual root model call."""
        self._validate_boundary(workflow, data)
        plan = workflow.plan(self.interaction.module, data)
        if expected_plan is not None and plan != expected_plan:
            raise RuntimeError("TDHook workflow plan changed after caller preflight")
        self._validate_gradient_contract(plan)
        event_start = len(self.interaction.events)
        with self.interaction, self.interaction.observe_module_calls():
            result = workflow.run(self.interaction.module, data)
        events = tuple(self.interaction.events[event_start:])
        record = WorkflowRunRecord(self.interaction.descriptor.identity, plan, events)
        if record.model_calls != plan.model_passes:
            raise RuntimeError(
                "TDHook workflow model-pass mismatch: "
                f"plan declares {plan.model_passes}, XDRL observed {record.model_calls}"
            )
        return TDHookWorkflowResult(result, record)

    def _validate_boundary(self, workflow: Workflow, data: TensorDictBase) -> None:
        if not isinstance(workflow, Workflow):
            raise TypeError(f"workflow must be a TDHook Workflow, got {type(workflow).__name__}")
        if not isinstance(data, TensorDictBase):
            raise TypeError(f"workflow data must be a TensorDict, got {type(data).__name__}")
        if _has_uninitialized_parameters(self.interaction.module):
            raise RuntimeError("interaction module has lazy parameters; call materialize() before workflow use")
        self.interaction.input_schema.validate_inputs(data)

    def _validate_gradient_contract(self, plan: WorkflowPlan) -> None:
        descriptor = self.interaction.descriptor
        for execution in plan.executions:
            if execution.gradient_mode is GradientMode.REQUIRED and (
                not descriptor.gradient_enabled or descriptor.inference_mode
            ):
                raise ValueError(
                    "gradient-required TDHook execution needs gradient_enabled=True and inference_mode=False"
                )
            if execution.gradient_mode is GradientMode.DISABLED and descriptor.gradient_enabled:
                raise ValueError("gradient-disabled TDHook execution is incompatible with gradient_enabled=True")


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


__all__ = ["TDHookWorkflowResult", "TDHookWorkflowRunner", "WorkflowRunRecord"]
