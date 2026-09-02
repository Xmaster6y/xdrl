"""The single XDRL entrypoint for native TDHook workflows."""

from __future__ import annotations

from tensordict import TensorDictBase
from tdhook.workflow import Workflow, WorkflowResult

from xdrl.interactions import Interaction


def run_workflow(interaction: Interaction, workflow: Workflow, data: TensorDictBase) -> WorkflowResult:
    """Run ``workflow`` against the interaction's unchanged module.

    TDHook owns planning, hooks, occurrences, cleanup, and the returned
    :class:`tdhook.workflow.WorkflowResult`. XDRL validates only the external
    TensorDict boundary and scopes the declared Torch execution modes.
    """

    if not isinstance(interaction, Interaction):
        raise TypeError(f"interaction must be an Interaction, got {type(interaction).__name__}")
    if not isinstance(workflow, Workflow):
        raise TypeError(f"workflow must be a TDHook Workflow, got {type(workflow).__name__}")
    interaction.validate_input(data)
    with interaction._execution_scope():
        result = workflow.run_with_plan(interaction.module, data)
    if not isinstance(result, WorkflowResult):
        raise TypeError("TDHook workflow returned an invalid result")
    interaction.validate_output(data, result.data)
    return result


__all__ = ["run_workflow"]
