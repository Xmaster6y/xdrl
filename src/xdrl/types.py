"""Small, native schemas for TensorDict interaction boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import torch
from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.data import Composite, TensorSpec


TensorDictKey: TypeAlias = NestedKey


class ModelRole(str, Enum):
    """The role a module plays in an RL system."""

    ACTOR = "actor"
    CRITIC = "critic"
    VALUE = "value"
    LOSS = "loss"
    ENCODER = "encoder"
    MIXER = "mixer"
    WORLD_MODEL = "world_model"


class KeyRole(str, Enum):
    """The semantic role of a TensorDict key."""

    STATE = "state"
    OBSERVATION = "observation"
    ACTION = "action"
    REWARD = "reward"
    TERMINATION = "termination"
    LOG_PROBABILITY = "log_probability"
    DISTRIBUTION_PARAMETER = "distribution_parameter"
    VALUE = "value"
    FEATURE = "feature"


class SchemaValidationError(ValueError):
    """A TensorDict does not satisfy an interaction schema."""


@dataclass(frozen=True, slots=True)
class BatchSemantics:
    """Names for the leading TensorDict batch dimensions, in order."""

    dimensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not dimension for dimension in self.dimensions):
            raise ValueError("batch dimension names must be non-empty")
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("batch dimension names must be unique")

    def validate(self, batch_size: torch.Size) -> None:
        if len(batch_size) != len(self.dimensions):
            raise SchemaValidationError(
                f"expected batch dimensions {self.dimensions!r}, got batch_size={tuple(batch_size)!r}"
            )


@dataclass(frozen=True, slots=True)
class KeySchema:
    """One required or optional TensorDict key with optional TorchRL constraints."""

    key: TensorDictKey
    role: KeyRole
    spec: TensorSpec | None = None
    required: bool = True

    def __post_init__(self) -> None:
        path = _key_path(self.key)
        if not path or any(not component for component in path):
            raise ValueError("schema keys must be non-empty")
        if type(self.required) is not bool:
            raise TypeError("required must be a boolean")


@dataclass(frozen=True, slots=True)
class TensorDictSchema:
    """A collection of semantic keys at one TensorDict boundary."""

    keys: tuple[KeySchema, ...]

    def __post_init__(self) -> None:
        paths = tuple(_key_path(entry.key) for entry in self.keys)
        if len(set(paths)) != len(paths):
            raise ValueError("schema keys must be unique")

    def validate(
        self,
        tensordict: TensorDictBase,
        batch: BatchSemantics = BatchSemantics(),
        *,
        boundary: str = "interaction",
    ) -> None:
        """Validate one input or output boundary without changing its TensorDict."""

        if not isinstance(tensordict, TensorDictBase):
            raise SchemaValidationError(f"{boundary} must be a TensorDict, got {type(tensordict).__name__}")
        batch.validate(tensordict.batch_size)
        missing = object()
        for entry in self.keys:
            value = tensordict.get(entry.key, missing)
            path = "/".join(_key_path(entry.key))
            if value is missing:
                if entry.required:
                    raise SchemaValidationError(f"{boundary} is missing key {path}")
                continue
            if not isinstance(value, (torch.Tensor, TensorDictBase)):
                raise SchemaValidationError(
                    f"{boundary} key {path} must contain a torch.Tensor or TensorDictBase, got {type(value).__name__}"
                )
            if entry.spec is not None and not _matches_spec(entry.spec, value, tensordict.batch_size):
                raise SchemaValidationError(
                    f"{boundary} key {path} does not satisfy {entry.spec!r}; got shape={tuple(value.shape)!r}"
                )

    def entry(self, key: TensorDictKey) -> KeySchema | None:
        """Return the declaration for ``key``, if present."""

        path = _key_path(key)
        return next((entry for entry in self.keys if _key_path(entry.key) == path), None)


def _key_path(key: TensorDictKey) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else tuple(key)


def _matches_spec(spec: TensorSpec, value: torch.Tensor | TensorDictBase, batch_size: torch.Size) -> bool:
    if isinstance(value, TensorDictBase):
        return isinstance(spec, Composite) and bool(spec.is_in(value))
    batch_dims = len(batch_size)
    return (
        value.shape[:batch_dims] == batch_size and value.shape[batch_dims:] == spec.shape and bool(spec.is_in(value))
    )


__all__ = [
    "BatchSemantics",
    "KeyRole",
    "KeySchema",
    "ModelRole",
    "SchemaValidationError",
    "TensorDictKey",
    "TensorDictSchema",
]
