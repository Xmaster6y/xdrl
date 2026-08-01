"""Typed execution contexts for individual TorchRL model invocations.

The persistent descriptor is deliberately tensor-free and serialisable.  The
runtime context owns the live module, representative batch, and temporary
execution state needed to invoke that module safely.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl.envs.utils import set_exploration_type

from xdrl.types import ModelRole, TensorDictKey, TensorDictSchema


class InteractionPhase(str, Enum):
    """Where an RL model invocation occurs."""

    COLLECTION = "collection"
    EVALUATION = "evaluation"
    REPLAY = "replay"
    LOSS = "loss"
    TARGET = "target"
    OPTIMISATION = "optimisation"


class LifecycleEventType(str, Enum):
    """Lifecycle state recorded without retaining model inputs or outputs."""

    BEFORE = "before"
    AFTER = "after"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class KeySnapshot:
    """Serialisable description of one declared TensorDict key."""

    path: tuple[str, ...]
    role: str
    presence: str
    feature_shape: tuple[int, ...] | None
    spec_type: str | None
    spec_constraints: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class SchemaSnapshot:
    """Serialisable projection of a :class:`TensorDictSchema`."""

    keys: tuple[KeySnapshot, ...]
    batch_dimensions: tuple[str, ...]

    @classmethod
    def from_schema(cls, schema: TensorDictSchema) -> SchemaSnapshot:
        return cls(
            keys=tuple(
                KeySnapshot(
                    path=_key_path(entry.key),
                    role=entry.role.value,
                    presence=entry.presence.value,
                    feature_shape=tuple(entry.spec.shape) if entry.spec is not None else None,
                    spec_type=type(entry.spec).__name__ if entry.spec is not None else None,
                    spec_constraints=_spec_constraints(entry.spec) if entry.spec is not None else None,
                )
                for entry in schema.keys
            ),
            batch_dimensions=schema.batch.dimensions,
        )


@dataclass(frozen=True, slots=True)
class InteractionDescriptor:
    """Persistent, tensor-free identity and semantics of an interaction.

    ``identity`` must be stable within one recorded run.  Events for a given
    identity are ordered by their monotonically increasing ``order`` field.
    """

    identity: str
    role: ModelRole
    phase: InteractionPhase
    module_path: str
    input_schema: SchemaSnapshot
    output_schema: SchemaSnapshot
    batch_dimensions: tuple[str, ...] = ()
    environment: str | None = None
    time_dimension: str | None = None
    agent_dimension: str | None = None
    objective: str | None = None
    recurrent_state_keys: tuple[tuple[str, ...], ...] = ()
    mask_keys: tuple[tuple[str, ...], ...] = ()
    exploration_mode: str | None = None
    stochastic_outputs: tuple[tuple[str, ...], ...] = ()
    gradient_enabled: bool = False
    inference_mode: bool = False
    autocast_device_type: str | None = None
    autocast_enabled: bool = False
    logical_step: int | None = None
    episode_id: str | int | None = None
    trajectory_id: str | int | None = None
    module_aliases: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with no tensors or modules."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """A tensor-free record of one attempted module invocation."""

    order: int
    kind: LifecycleEventType
    interaction_id: str
    phase: InteractionPhase
    module_path: str
    key_shapes: Mapping[str, tuple[int, ...] | None]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible event representation."""
        return asdict(self)


class HookContextFactory(Protocol):
    """Create a context manager that installs temporary TDHook state."""

    def __call__(self) -> Any: ...


@dataclass(slots=True)
class RuntimeInteractionContext:
    """Ephemeral execution wrapper around one existing TensorDict module.

    Construction validates the supplied representative input.  ``invoke``
    validates the live input/output around the actual module call and records
    lifecycle metadata.  The module itself, tensors, and hook state never
    enter :class:`InteractionDescriptor` or :class:`LifecycleEvent`.
    """

    descriptor: InteractionDescriptor
    module: TensorDictModuleBase
    input_schema: TensorDictSchema
    output_schema: TensorDictSchema
    representative_input: TensorDictBase
    hook_context_factory: HookContextFactory | None = None
    events: list[LifecycleEvent] = field(default_factory=list, init=False)
    _stack: ExitStack | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if SchemaSnapshot.from_schema(self.input_schema) != self.descriptor.input_schema:
            raise ValueError("input_schema does not match the interaction descriptor snapshot")
        if SchemaSnapshot.from_schema(self.output_schema) != self.descriptor.output_schema:
            raise ValueError("output_schema does not match the interaction descriptor snapshot")
        self.input_schema.validate_inputs(self.representative_input)

    def __enter__(self) -> RuntimeInteractionContext:
        if self._stack is not None:
            raise RuntimeError("interaction context is already active")
        stack = ExitStack()
        try:
            if self.descriptor.exploration_mode is not None:
                stack.enter_context(set_exploration_type(self.descriptor.exploration_mode))
            stack.enter_context(torch.inference_mode(self.descriptor.inference_mode))
            if not self.descriptor.inference_mode:
                stack.enter_context(torch.set_grad_enabled(self.descriptor.gradient_enabled))
            if self.descriptor.autocast_device_type is not None:
                stack.enter_context(
                    torch.autocast(
                        device_type=self.descriptor.autocast_device_type,
                        enabled=self.descriptor.autocast_enabled,
                    )
                )
            if self.hook_context_factory is not None:
                stack.enter_context(self.hook_context_factory())
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        return self

    def __exit__(self, *exc_info: object) -> bool | None:
        if self._stack is None:
            return None
        stack, self._stack = self._stack, None
        return stack.__exit__(*exc_info)

    def invoke(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Invoke the wrapped module and append before/after/failure events."""
        if self._stack is None:
            raise RuntimeError("invoke must be called inside the interaction context")
        self._record(LifecycleEventType.BEFORE, tensordict)
        try:
            self.input_schema.validate_inputs(tensordict)
            result = self.module(tensordict)
        except BaseException as error:
            self._record(LifecycleEventType.FAILURE, tensordict, error)
            raise
        try:
            self.output_schema.validate_outputs(result)
        except BaseException as error:
            self._record(LifecycleEventType.FAILURE, result, error)
            raise
        self._record(LifecycleEventType.AFTER, result)
        return result

    def _record(
        self, kind: LifecycleEventType, tensordict: TensorDictBase, error: BaseException | None = None
    ) -> None:
        self.events.append(
            LifecycleEvent(
                order=len(self.events),
                kind=kind,
                interaction_id=self.descriptor.identity,
                phase=self.descriptor.phase,
                module_path=self.descriptor.module_path,
                key_shapes=_key_shapes(tensordict),
                error=f"{type(error).__name__}: {error}" if error is not None else None,
            )
        )


def _key_path(key: TensorDictKey) -> tuple[str, ...]:
    return tuple(str(part) for part in key) if isinstance(key, tuple) else (str(key),)


def _key_shapes(tensordict: TensorDictBase) -> dict[str, tuple[int, ...] | None]:
    """Capture only key paths and shapes, never references to tensor values."""
    result: dict[str, tuple[int, ...] | None] = {}
    for key, value in tensordict.items(include_nested=True, leaves_only=True):
        path = "/".join(_key_path(key))
        result[path] = tuple(value.shape) if isinstance(value, torch.Tensor) else None
    return result


def _spec_constraints(spec: Any) -> dict[str, Any]:
    """Project TensorSpec validation constraints into JSON-compatible values."""
    constraints: dict[str, Any] = {}
    for name in ("dtype", "device", "domain", "low", "high", "n", "nvec", "mask"):
        if hasattr(spec, name):
            constraints[name] = _json_value(getattr(spec, name))
    return constraints


def _json_value(value: Any) -> Any:
    """Convert scalar spec metadata and tensors without retaining tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, torch.Size):
        return tuple(value)
    if isinstance(value, tuple):
        return tuple(_json_value(item) for item in value)
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value
