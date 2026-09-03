"""The single XDRL entrypoint for native TDHook workflows."""

from __future__ import annotations

import torch
from tensordict import TensorDictBase
from tdhook.execution import GradientMode
from tdhook.workflow import Workflow, WorkflowPlan, WorkflowResult

from xdrl.interactions import Interaction


def run_workflow(interaction: Interaction, workflow: Workflow, data: TensorDictBase) -> WorkflowResult:
    """Run ``workflow`` against the interaction's unchanged module.

    TDHook owns planning, hooks, occurrences, cleanup, and the returned
    :class:`tdhook.workflow.WorkflowResult`. XDRL validates only the external
    TensorDict boundary. The caller owns the surrounding Torch execution state.
    """

    if not isinstance(interaction, Interaction):
        raise TypeError(f"interaction must be an Interaction, got {type(interaction).__name__}")
    if not isinstance(workflow, Workflow):
        raise TypeError(f"workflow must be a TDHook Workflow, got {type(workflow).__name__}")
    interaction.validate_input(data)
    plan = workflow.plan(interaction.module, data)
    _validate_execution_modes(plan)
    result = workflow.run_with_plan(interaction.module, data)
    if not isinstance(result, WorkflowResult):
        raise TypeError("TDHook workflow returned an invalid result")
    interaction.validate_output(data, result.data)
    return result


def _validate_execution_modes(plan: WorkflowPlan) -> None:
    for execution in plan.executions:
        if execution.gradient_mode is GradientMode.REQUIRED and (
            torch.is_inference_mode_enabled() or not torch.is_grad_enabled()
        ):
            raise ValueError("gradient-required TDHook execution requires enabled autograd outside inference mode")
        if execution.gradient_mode is GradientMode.DISABLED and torch.is_grad_enabled():
            raise ValueError("gradient-disabled TDHook execution requires a no-grad context")


__all__ = ["run_workflow"]
