"""Typed, bounded observation traces for RL model interactions.

The trace owns optional tensor snapshots, while every observation record is
serialisable even when its payload is omitted.  It intentionally does not
implement probes or attribution algorithms; those consumers can subscribe to
the stream or operate on retained records.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

import torch
from tensordict import TensorDictBase

from xdrl.interactions import InteractionDescriptor
from xdrl.types import KeyRole, TensorDictKey


class ObservationKind(str, Enum):
    """Semantic category of a captured value."""

    MODULE_INPUT = "module_input"
    MODULE_OUTPUT = "module_output"
    ACTIVATION = "activation"
    GRADIENT = "gradient"
    DISTRIBUTION_PARAMETER = "distribution_parameter"
    ACTION = "action"
    VALUE = "value"
    TENSORDICT_KEY = "tensordict_key"


class HookDirection(str, Enum):
    """Whether a hook observed a module's input or output."""

    INPUT = "input"
    OUTPUT = "output"
    TENSOR = "tensor"


class TensorRetention(str, Enum):
    """How an observation record retains its optional tensor payload."""

    METADATA = "metadata"
    DETACHED = "detached"
    CPU = "cpu"


class OverflowPolicy(str, Enum):
    """Action taken when the in-memory trace reaches its record capacity."""

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Explicit graph, device, sampling, reduction, and backpressure policy.

    ``every_n`` samples observations by monotonically increasing observation
    order. ``reduction`` applies only across named batch dimensions and emits
    a record with those dimensions removed.  Payloads are always detached and
    cloned, so this package never retains a computation graph by accident.
    """

    tensor: TensorRetention = TensorRetention.METADATA
    every_n: int = 1
    reduction: str | None = None
    max_records: int = 1_024
    overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST

    def __post_init__(self) -> None:
        if self.every_n < 1:
            raise ValueError("every_n must be at least 1")
        if self.max_records < 0:
            raise ValueError("max_records must be non-negative")
        if self.reduction not in {None, "mean", "sum", "max"}:
            raise ValueError("reduction must be one of None, 'mean', 'sum', or 'max'")


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    """One observation and its tensor-free interpretation context."""

    order: int
    interaction_id: str
    phase: str
    module_path: str
    model_role: str
    model_id: str | None
    checkpoint_id: str | None
    exploration_mode: str | None
    kind: ObservationKind
    hook_direction: HookDirection
    target: str
    key_path: tuple[str, ...] | None
    batch_dimensions: tuple[str, ...]
    time_dimension: str | None
    agent_dimension: str | None
    logical_step: int | None
    episode_id: str | int | None
    trajectory_id: str | int | None
    shape: tuple[int, ...]
    dtype: str
    device: str
    retained_batch_dimensions: tuple[str, ...]
    payload: torch.Tensor | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record without its optional tensor payload."""
        return {item.name: getattr(self, item.name) for item in fields(self) if item.name != "payload"}


ObservationCallback = Callable[[ObservationRecord], None]


@dataclass(slots=True)
class ObservationTrace:
    """A bounded trace that can retain tensors and/or stream typed records."""

    policy: RetentionPolicy = field(default_factory=RetentionPolicy)
    callback: ObservationCallback | None = None
    records: deque[ObservationRecord] = field(default_factory=deque, init=False)
    dropped: int = field(default=0, init=False)
    _seen: int = field(default=0, init=False, repr=False)

    def observe_tensor(
        self,
        descriptor: InteractionDescriptor,
        tensor: torch.Tensor,
        *,
        kind: ObservationKind,
        target: str,
        direction: HookDirection = HookDirection.TENSOR,
        key: TensorDictKey | None = None,
        batch_dimensions: tuple[str, ...] | None = None,
    ) -> ObservationRecord | None:
        """Capture a hook, gradient, or arbitrary tensor with explicit retention.

        Pass ``batch_dimensions`` only when the leading dimensions of ``tensor``
        are known to have those semantics.  This prevents an arbitrary hook or
        gradient tensor from being reduced along unrelated feature dimensions.
        """
        order = self._seen
        self._seen += 1
        if order % self.policy.every_n:
            return None
        value, retained_dimensions = _retain(tensor, batch_dimensions, self.policy)
        record = ObservationRecord(
            order=order,
            interaction_id=descriptor.identity,
            phase=descriptor.phase.value,
            module_path=descriptor.module_path,
            model_role=descriptor.role.value,
            model_id=descriptor.model_id,
            checkpoint_id=descriptor.checkpoint_id,
            exploration_mode=descriptor.exploration_mode,
            kind=kind,
            hook_direction=direction,
            target=target,
            key_path=_key_path(key) if key is not None else None,
            batch_dimensions=descriptor.batch_dimensions,
            time_dimension=descriptor.time_dimension,
            agent_dimension=descriptor.agent_dimension,
            logical_step=descriptor.logical_step,
            episode_id=descriptor.episode_id,
            trajectory_id=descriptor.trajectory_id,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            retained_batch_dimensions=retained_dimensions,
            payload=value,
        )
        self._append(record)
        if self.callback is not None:
            self.callback(record)
        return record

    def capture_tensordict(
        self,
        descriptor: InteractionDescriptor,
        tensordict: TensorDictBase,
        *,
        direction: HookDirection,
        roles: Mapping[tuple[str, ...], KeyRole],
    ) -> tuple[ObservationRecord, ...]:
        """Capture all tensor leaves while preserving nested TensorDict keys."""
        records = []
        for key, value in tensordict.items(include_nested=True, leaves_only=True):
            if not isinstance(value, torch.Tensor):
                continue
            path = _key_path(key)
            # TensorDict modules commonly mutate their input in place.  Only
            # declared keys belong to this interaction direction; retaining
            # untouched input leaves as module outputs would mislabel them.
            # A declared TensorDict/Composite parent applies to all its leaves.
            role = _role_for_path(path, roles)
            if role is None:
                continue
            records.append(
                self.observe_tensor(
                    descriptor,
                    value,
                    kind=_kind_for(role, direction),
                    target="/".join(path),
                    direction=direction,
                    key=key,
                    batch_dimensions=descriptor.batch_dimensions,
                )
            )
        return tuple(record for record in records if record is not None)

    def _append(self, record: ObservationRecord) -> None:
        if self.policy.max_records == 0 or len(self.records) >= self.policy.max_records:
            if self.policy.overflow is OverflowPolicy.DROP_NEWEST or self.policy.max_records == 0:
                self.dropped += 1
                return
            if self.policy.overflow is OverflowPolicy.RAISE:
                raise BufferError("observation trace reached max_records")
            self.records.popleft()
            self.dropped += 1
        self.records.append(record)


def _retain(
    tensor: torch.Tensor, batch_dimensions: tuple[str, ...] | None, policy: RetentionPolicy
) -> tuple[torch.Tensor | None, tuple[str, ...]]:
    known_dimensions = () if batch_dimensions is None else batch_dimensions
    if policy.tensor is TensorRetention.METADATA:
        return None, known_dimensions
    value = tensor.detach()
    retained_dimensions = known_dimensions
    if policy.reduction is not None and batch_dimensions:
        dims = tuple(range(len(batch_dimensions)))
        value = (
            torch.amax(value, dim=dims) if policy.reduction == "max" else getattr(value, policy.reduction)(dim=dims)
        )
        retained_dimensions = ()
    if policy.tensor is TensorRetention.CPU:
        value = value.cpu()
    return value.clone(), retained_dimensions


def _kind_for(role: KeyRole | None, direction: HookDirection) -> ObservationKind:
    if role is KeyRole.ACTION:
        return ObservationKind.ACTION
    if role is KeyRole.VALUE:
        return ObservationKind.VALUE
    if role is KeyRole.DISTRIBUTION_PARAMETER:
        return ObservationKind.DISTRIBUTION_PARAMETER
    return ObservationKind.MODULE_INPUT if direction is HookDirection.INPUT else ObservationKind.MODULE_OUTPUT


def _role_for_path(path: tuple[str, ...], roles: Mapping[tuple[str, ...], KeyRole]) -> KeyRole | None:
    """Find the most specific declared schema key that contains ``path``."""
    for length in range(len(path), 0, -1):
        role = roles.get(path[:length])
        if role is not None:
            return role
    return None


def _key_path(key: TensorDictKey) -> tuple[str, ...]:
    return tuple(str(part) for part in key) if isinstance(key, tuple) else (str(key),)
