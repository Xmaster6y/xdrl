"""TorchRL-native contracts for RL model roles and TensorDict schemas.

The types in this module describe interfaces around TorchRL objects.  They do
not introduce a container or a spec hierarchy: data remains a
``TensorDictBase`` and value constraints remain TorchRL ``TensorSpec`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, TypeAlias, runtime_checkable

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey
from torchrl.data import TensorSpec

TensorDictKey: TypeAlias = NestedKey
"""A TensorDict key, including nested paths such as ``("agents", "action")``."""


class ModelRole(str, Enum):
    """The public role a TorchRL module plays in an RL system."""

    ACTOR = "actor"
    CRITIC = "critic"
    VALUE = "value"
    LOSS = "loss"
    ENCODER = "encoder"
    MIXER = "mixer"
    WORLD_MODEL = "world_model"


class KeyRole(str, Enum):
    """Semantic role of a key without changing its TensorDict representation."""

    STATE = "state"
    OBSERVATION = "observation"
    ACTION = "action"
    REWARD = "reward"
    TERMINATION = "termination"
    LOG_PROBABILITY = "log_probability"
    DISTRIBUTION_PARAMETER = "distribution_parameter"
    VALUE = "value"
    FEATURE = "feature"


class KeyPresence(str, Enum):
    """Whether a contract consumes, produces, or optionally uses a key."""

    REQUIRED = "required"
    PRODUCED = "produced"
    OPTIONAL = "optional"


@dataclass(frozen=True, slots=True)
class BatchSemantics:
    """Meaning of leading TensorDict dimensions, independent of feature shape.

    ``dimensions`` names every leading dimension in order.  Feature dimensions
    are represented only by a key's TorchRL ``TensorSpec``.
    """

    dimensions: tuple[str, ...] = ()

    def validate(self, batch_size: torch.Size) -> None:
        """Raise when a TensorDict does not have the declared number of batch dims."""
        if len(batch_size) != len(self.dimensions):
            raise SchemaValidationError(
                "batch dimensions mismatch: "
                f"schema declares {self.dimensions!r}, TensorDict has batch_size={tuple(batch_size)!r}"
            )


@dataclass(frozen=True, slots=True)
class KeySchema:
    """A semantic TensorDict key refined by an optional TorchRL spec."""

    key: TensorDictKey
    role: KeyRole
    presence: KeyPresence
    spec: TensorSpec | None = None


class SchemaValidationError(ValueError):
    """A TensorDict violates a model schema contract."""


_MISSING = object()


@dataclass(frozen=True, slots=True)
class TensorDictSchema:
    """I/O schema for a role, reusing nested keys and TorchRL specs directly."""

    keys: tuple[KeySchema, ...]
    batch: BatchSemantics = field(default_factory=BatchSemantics)

    def __post_init__(self) -> None:
        duplicates = [entry.key for entry in self.keys if sum(item.key == entry.key for item in self.keys) > 1]
        if duplicates:
            raise ValueError(f"schema contains duplicate keys: {duplicates!r}")

    def validate_inputs(self, tensordict: TensorDictBase) -> None:
        """Validate required and optional consumed keys in ``tensordict``."""
        self._validate(tensordict, {KeyPresence.REQUIRED, KeyPresence.OPTIONAL})

    def validate_outputs(self, tensordict: TensorDictBase) -> None:
        """Validate produced keys in ``tensordict`` after a module call."""
        self._validate(tensordict, {KeyPresence.PRODUCED})

    def _validate(self, tensordict: TensorDictBase, presences: set[KeyPresence]) -> None:
        self.batch.validate(tensordict.batch_size)
        for entry in self.keys:
            if entry.presence not in presences:
                continue
            value = tensordict.get(entry.key, _MISSING)
            path = _display_key(entry.key)
            if value is _MISSING:
                if entry.presence is KeyPresence.REQUIRED or entry.presence is KeyPresence.PRODUCED:
                    raise SchemaValidationError(f"missing {entry.presence.value} key at {path}")
                continue
            if not isinstance(value, torch.Tensor):
                raise SchemaValidationError(f"key at {path} must contain a torch.Tensor, got {type(value).__name__}")
            if entry.spec is not None and not _matches_spec(entry.spec, value):
                raise SchemaValidationError(
                    f"spec mismatch at {path}: got shape={tuple(value.shape)!r}, expected feature shape="
                    f"{tuple(entry.spec.shape)!r} and spec={entry.spec!r}"
                )


@runtime_checkable
class ContractModule(Protocol):
    """TorchRL module interface annotated with explicit input and output schemas."""

    role: ModelRole
    input_schema: TensorDictSchema
    output_schema: TensorDictSchema

    def __call__(self, tensordict: TensorDictBase, *args: object, **kwargs: object) -> TensorDictBase: ...


def validate_module(module: ContractModule, tensordict: TensorDictBase) -> TensorDictBase:
    """Validate a contract module's input and output around one TensorDict call.

    ``TensorDictModuleBase`` is deliberately not required: probabilistic actors,
    loss modules, and user modules may expose compatible TensorDict call
    interfaces without sharing that concrete base class.
    """
    module.input_schema.validate_inputs(tensordict)
    result = module(tensordict)
    module.output_schema.validate_outputs(result)
    return result


TorchRLModule: TypeAlias = TensorDictModuleBase
"""Concrete TorchRL TensorDict module base, re-exported for type annotations."""


def _display_key(key: TensorDictKey) -> str:
    return "/".join(key) if isinstance(key, tuple) else key


def _matches_spec(spec: TensorSpec, value: torch.Tensor) -> bool:
    """Check feature shape before applying a TorchRL spec membership constraint."""
    feature_shape = spec.shape
    if feature_shape and (len(value.shape) < len(feature_shape) or value.shape[-len(feature_shape) :] != feature_shape):
        return False
    return bool(spec.is_in(value))
