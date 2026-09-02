"""One typed execution boundary around an existing TorchRL module."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl.envs.utils import set_exploration_type

from xdrl.types import BatchSemantics, KeyRole, ModelRole, TensorDictKey, TensorDictSchema


@dataclass(frozen=True, slots=True)
class RecurrentStateTransition:
    """A state key consumed now and its corresponding next-state key."""

    input_key: TensorDictKey
    output_key: TensorDictKey


@dataclass(frozen=True, slots=True)
class RecurrentSemantics:
    """The recurrent state and reset keys that affect boundary validation."""

    transitions: tuple[RecurrentStateTransition, ...]
    reset_keys: tuple[TensorDictKey, ...] = ()

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


@dataclass(frozen=True, slots=True)
class InteractionSpec:
    """The complete static contract for one TorchRL module call."""

    role: ModelRole
    inputs: TensorDictSchema
    outputs: TensorDictSchema
    batch: BatchSemantics = BatchSemantics()
    recurrent: RecurrentSemantics | None = None
    training: bool | None = None
    gradient_enabled: bool = False
    inference_mode: bool = False
    exploration_mode: str | None = None
    autocast_device_type: str | None = None
    autocast_enabled: bool = False

    def __post_init__(self) -> None:
        if self.inference_mode and self.gradient_enabled:
            raise ValueError("inference_mode and gradient_enabled cannot both be enabled")
        if self.autocast_enabled and self.autocast_device_type is None:
            raise ValueError("autocast_enabled requires autocast_device_type")
        if self.training is not None and type(self.training) is not bool:
            raise TypeError("training must be a boolean or None")
        if self.recurrent is not None:
            _validate_recurrent_spec(self)


@dataclass(slots=True)
class Interaction:
    """Validate and execute one unchanged TensorDict module.

    XDRL owns only this boundary. TensorDict and TorchRL own the data and model;
    TDHook owns model-internal hooks, targets, and workflow execution.
    """

    module: TensorDictModuleBase
    spec: InteractionSpec

    def __post_init__(self) -> None:
        if not isinstance(self.module, TensorDictModuleBase):
            raise TypeError("Interaction requires a TensorDictModuleBase")

    def __call__(self, data: TensorDictBase) -> TensorDictBase:
        """Validate and execute the module once."""

        return self._invoke(data, self.module)

    def validate_input(self, data: TensorDictBase) -> None:
        self.spec.inputs.validate(data, self.spec.batch, boundary="interaction input")
        recurrent = self.spec.recurrent
        if recurrent is None:
            return
        for key in recurrent.reset_keys:
            reset = data.get(key)
            if not isinstance(reset, torch.Tensor) or reset.dtype is not torch.bool:
                raise ValueError(f"recurrent reset key {'/'.join(_key_path(key))} must contain a boolean tensor")

    def validate_output(self, inputs: TensorDictBase, outputs: TensorDictBase) -> None:
        self.spec.outputs.validate(outputs, self.spec.batch, boundary="interaction output")
        recurrent = self.spec.recurrent
        if recurrent is None:
            return
        for transition in recurrent.transitions:
            previous = inputs.get(transition.input_key)
            following = outputs.get(transition.output_key)
            if not isinstance(previous, torch.Tensor) or not isinstance(following, torch.Tensor):
                raise ValueError("recurrent state transitions must connect tensor-valued keys")
            if previous.shape != following.shape or previous.dtype != following.dtype:
                raise ValueError(
                    f"recurrent state transition {'/'.join(_key_path(transition.input_key))} -> "
                    f"{'/'.join(_key_path(transition.output_key))} changed shape or dtype"
                )

    def _invoke(
        self,
        data: TensorDictBase,
        operation: Callable[[TensorDictBase], TensorDictBase],
    ) -> TensorDictBase:
        self.validate_input(data)
        with self._execution_scope():
            result = operation(data)
        if not isinstance(result, TensorDictBase):
            raise TypeError(f"interaction module must return a TensorDict, got {type(result).__name__}")
        self.validate_output(data, result)
        return result

    @contextmanager
    def _execution_scope(self) -> Iterator[None]:
        with ExitStack() as stack:
            if self.spec.training is not None:
                states = tuple((module, module.training) for module in self.module.modules())
                self.module.train(self.spec.training)
                stack.callback(_restore_training_states, states)
            if self.spec.exploration_mode is not None:
                stack.enter_context(set_exploration_type(self.spec.exploration_mode))
            stack.enter_context(torch.inference_mode(self.spec.inference_mode))
            if not self.spec.inference_mode:
                stack.enter_context(torch.set_grad_enabled(self.spec.gradient_enabled))
            if self.spec.autocast_device_type is not None:
                stack.enter_context(
                    torch.autocast(
                        device_type=self.spec.autocast_device_type,
                        enabled=self.spec.autocast_enabled,
                    )
                )
            yield


def _validate_recurrent_spec(spec: InteractionSpec) -> None:
    recurrent = spec.recurrent
    assert recurrent is not None
    for transition in recurrent.transitions:
        input_entry = spec.inputs.entry(transition.input_key)
        output_entry = spec.outputs.entry(transition.output_key)
        if input_entry is None or input_entry.role is not KeyRole.STATE or not input_entry.required:
            raise ValueError("recurrent input keys must be required state inputs")
        if output_entry is None or output_entry.role is not KeyRole.STATE or not output_entry.required:
            raise ValueError("recurrent output keys must be required state outputs")
    for reset_key in recurrent.reset_keys:
        if spec.inputs.entry(reset_key) is None:
            raise ValueError(f"recurrent reset key {'/'.join(_key_path(reset_key))} is not an input")


def _restore_training_states(states: tuple[tuple[torch.nn.Module, bool], ...]) -> None:
    for module, training in states:
        module.training = training


def _key_path(key: TensorDictKey) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else tuple(key)


__all__ = [
    "Interaction",
    "InteractionSpec",
    "RecurrentSemantics",
    "RecurrentStateTransition",
]
