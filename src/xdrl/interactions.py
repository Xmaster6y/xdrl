"""RL semantics around one native TorchRL module call."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey


@dataclass(frozen=True, slots=True)
class RecurrentStateTransition:
    """A recurrent input key and the output key containing its next value."""

    input_key: NestedKey
    output_key: NestedKey


@dataclass(frozen=True, slots=True)
class RecurrentSemantics:
    """Recurrent state transitions and reset keys for one module call."""

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


@dataclass(slots=True)
class Interaction:
    """Validate the RL boundary of an unchanged TensorDict module.

    The module remains the source of truth for required input keys, produced
    output keys, and any TorchRL ``SafeModule`` specs.
    """

    module: TensorDictModuleBase
    recurrent: RecurrentSemantics | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.module, TensorDictModuleBase):
            raise TypeError("Interaction requires a TensorDictModuleBase")
        if self.recurrent is not None:
            _validate_recurrent_module(self.module, self.recurrent)

    def __call__(self, data: TensorDictBase) -> TensorDictBase:
        """Run the module once under the caller's native Torch state."""

        self.validate_input(data)
        result = self.module(data)
        self.validate_output(data, result)
        return result

    def validate_input(self, data: TensorDictBase) -> None:
        if not isinstance(data, TensorDictBase):
            raise TypeError(f"interaction input must be a TensorDict, got {type(data).__name__}")
        if self.recurrent is not None:
            for key in self.recurrent.reset_keys:
                reset = data.get(key)
                if not isinstance(reset, torch.Tensor) or reset.dtype is not torch.bool:
                    raise ValueError(f"recurrent reset key {'/'.join(_key_path(key))} must contain a boolean tensor")

    def validate_output(self, inputs: TensorDictBase, outputs: TensorDictBase) -> None:
        if not isinstance(outputs, TensorDictBase):
            raise TypeError(f"interaction module must return a TensorDict, got {type(outputs).__name__}")
        if self.recurrent is not None:
            for transition in self.recurrent.transitions:
                previous = inputs.get(transition.input_key)
                following = outputs.get(transition.output_key)
                if not isinstance(previous, torch.Tensor) or not isinstance(following, torch.Tensor):
                    raise ValueError("recurrent state transitions must connect tensor-valued keys")
                if previous.shape != following.shape or previous.dtype != following.dtype:
                    raise ValueError(
                        f"recurrent state transition {'/'.join(_key_path(transition.input_key))} -> "
                        f"{'/'.join(_key_path(transition.output_key))} changed shape or dtype"
                    )


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


def _key_path(key: NestedKey) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else tuple(key)


__all__ = ["Interaction", "RecurrentSemantics", "RecurrentStateTransition"]
