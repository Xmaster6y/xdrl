"""Interpret existing TorchRL modules without restating their configuration."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from tdhook.execution import GradientMode
from tdhook.workflow import Workflow, WorkflowPlan, WorkflowResult
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey

if TYPE_CHECKING:
    from xdrl.modules import ModuleInterpretation
    from xdrl.objectives import ObjectiveInterpretation


@dataclass(frozen=True, slots=True)
class RecurrentStateTransition:
    """A recurrent input key and the output key containing its next value."""

    input_key: NestedKey
    output_key: NestedKey


@dataclass(frozen=True, slots=True)
class RecurrentSemantics:
    """Recurrent state transitions and reset keys for one TorchRL module call."""

    transitions: tuple[RecurrentStateTransition, ...]
    reset_keys: tuple[NestedKey, ...] = ()

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("recurrent semantics require at least one state transition")
        input_paths = tuple(_key_path(item.input_key) for item in self.transitions)
        output_paths = tuple(_key_path(item.output_key) for item in self.transitions)
        reset_paths = tuple(_key_path(key) for key in self.reset_keys)
        if len(set(input_paths)) != len(input_paths) or len(set(output_paths)) != len(output_paths):
            raise ValueError("recurrent state transition keys must be unique")
        if len(set(reset_paths)) != len(reset_paths):
            raise ValueError("recurrent reset keys must be unique")

    @classmethod
    def from_torchrl(cls, *state_keys: NestedKey, reset_key: NestedKey = "is_init") -> RecurrentSemantics:
        """Use TorchRL's ``next`` and ``is_init`` recurrent conventions."""

        return cls(
            tuple(RecurrentStateTransition(key, ("next", *_key_path(key))) for key in state_keys),
            reset_keys=(reset_key,),
        )


@dataclass(frozen=True, slots=True)
class Component:
    """An executable view of a network already owned by a TorchRL object.

    ``params`` is the functional parameter TensorDict held by a TorchRL loss.
    XDRL temporarily materializes it on ``module`` for a direct call or TDHook
    workflow and lets TensorDict restore the prior module state afterwards.
    """

    module: TensorDictModuleBase
    name: str = "module"
    params: TensorDictBase | None = None
    recurrent: RecurrentSemantics | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.module, TensorDictModuleBase):
            raise TypeError("a component requires a TensorDictModuleBase")
        if not self.name:
            raise ValueError("a component name must be non-empty")
        if self.recurrent is not None:
            _validate_recurrent_module(self.module, self.recurrent)

    def __call__(self, data: TensorDictBase) -> TensorDictBase:
        """Execute this component with its online or target parameters."""

        self.validate_input(data)
        with self.parameter_context():
            result = self.module(data)
        self.validate_output(data, result)
        return result

    def run(self, workflow: Workflow, data: TensorDictBase) -> WorkflowResult:
        """Run a native TDHook workflow against this TorchRL component."""

        if not isinstance(workflow, Workflow):
            raise TypeError(f"workflow must be a TDHook Workflow, got {type(workflow).__name__}")
        self.validate_input(data)
        with self.parameter_context():
            plan = workflow.plan(self.module, data)
            _validate_execution_modes(plan)
            result = workflow.run_with_plan(self.module, data)
        if not isinstance(result, WorkflowResult):
            raise TypeError("TDHook workflow returned an invalid result")
        self.validate_output(data, result.data)
        return result

    def with_recurrent(self, recurrent: RecurrentSemantics) -> Component:
        """Return this component with recurrent call semantics attached."""

        return Component(self.module, self.name, self.params, recurrent)

    def parameter_context(self) -> AbstractContextManager[Any]:
        """Materialize this view's functional parameters for one bounded call."""

        if self.params is None:
            return nullcontext()
        to_module = getattr(self.params, "to_module", None)
        if to_module is None:
            raise TypeError(f"parameters for {self.name!r} do not support TensorDict.to_module")
        return to_module(self.module, preserve_module_state=True)

    def validate_input(self, data: TensorDictBase) -> None:
        if not isinstance(data, TensorDictBase):
            raise TypeError(f"component input must be a TensorDict, got {type(data).__name__}")
        if self.recurrent is not None:
            for key in self.recurrent.reset_keys:
                reset = data.get(key)
                if not isinstance(reset, torch.Tensor) or reset.dtype is not torch.bool:
                    raise ValueError(f"recurrent reset key {'/'.join(_key_path(key))} must contain a boolean tensor")

    def validate_output(self, inputs: TensorDictBase, outputs: TensorDictBase) -> None:
        if not isinstance(outputs, TensorDictBase):
            raise TypeError(f"component module must return a TensorDict, got {type(outputs).__name__}")
        if self.recurrent is not None:
            for transition in self.recurrent.transitions:
                previous = inputs.get(transition.input_key)
                following = outputs.get(transition.output_key)
                if not isinstance(previous, torch.Tensor) or not isinstance(following, torch.Tensor):
                    raise TypeError("recurrent state transitions must connect tensor-valued keys")
                if previous.shape != following.shape or previous.dtype != following.dtype:
                    raise ValueError(
                        f"recurrent state transition {'/'.join(_key_path(transition.input_key))} -> "
                        f"{'/'.join(_key_path(transition.output_key))} changed shape or dtype"
                    )


def interpret(
    subject: object,
    *,
    recurrent: RecurrentSemantics | None = None,
) -> Component | ModuleInterpretation | ObjectiveInterpretation:
    """Return the native XDRL view for a TorchRL module or supported loss.

    Plain TensorDict modules become :class:`Component` objects. Supported
    TorchRL loss modules are dispatched to explicit algorithm integrations.
    """

    if isinstance(subject, TensorDictModuleBase):
        from torchrl.objectives import LossModule

        if isinstance(subject, LossModule):
            if recurrent is not None:
                raise TypeError("recurrent semantics belong to a selected loss component, not the loss itself")
            from xdrl.objectives import interpret_objective

            return interpret_objective(subject)
        if recurrent is not None:
            return Component(subject, recurrent=recurrent)
        from xdrl.modules import interpret_module

        return interpret_module(subject)
    raise TypeError(f"cannot interpret {type(subject).__name__}; expected a TorchRL TensorDict module or loss")


def _validate_recurrent_module(module: TensorDictModuleBase, recurrent: RecurrentSemantics) -> None:
    input_paths = {_key_path(key) for key in module.in_keys}
    output_paths = {_key_path(key) for key in module.out_keys}
    for transition in recurrent.transitions:
        if _key_path(transition.input_key) not in input_paths:
            raise ValueError(f"recurrent state key {'/'.join(_key_path(transition.input_key))} is not a module input")
        if _key_path(transition.output_key) not in output_paths:
            raise ValueError(
                f"recurrent next-state key {'/'.join(_key_path(transition.output_key))} is not a module output"
            )
    for key in recurrent.reset_keys:
        if _key_path(key) not in input_paths:
            raise ValueError(f"recurrent reset key {'/'.join(_key_path(key))} is not a module input")


def _validate_execution_modes(plan: WorkflowPlan) -> None:
    for execution in plan.executions:
        if execution.gradient_mode is GradientMode.REQUIRED and (
            torch.is_inference_mode_enabled() or not torch.is_grad_enabled()
        ):
            raise ValueError("gradient-required TDHook execution requires enabled autograd outside inference mode")
        if execution.gradient_mode is GradientMode.DISABLED and torch.is_grad_enabled():
            raise ValueError("gradient-disabled TDHook execution requires a no-grad context")


def _key_path(key: NestedKey) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else tuple(key)


__all__ = ["Component", "RecurrentSemantics", "RecurrentStateTransition", "interpret"]
