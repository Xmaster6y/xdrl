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
from typing import TYPE_CHECKING, Any, Callable, Protocol

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl.envs.utils import set_exploration_type

from xdrl.interventions import InterventionTiming
from xdrl.types import KeyPresence, KeyRole, ModelRole, TensorDictKey, TensorDictSchema

if TYPE_CHECKING:
    from xdrl.interventions import InterventionController
    from xdrl.observations import ObservationTrace


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


class InteractionTopology(str, Enum):
    """How a multi-agent module relates parameters and semantic agents."""

    INDEPENDENT = "independent"
    PARAMETER_SHARED = "parameter_shared"
    CENTRALISED_CRITIC = "centralised_critic"
    MIXER = "mixer"


class RecurrentCollectorMode(str, Enum):
    """Collector execution modes whose recurrent state lifecycle is known."""

    DIRECT = "direct"
    SYNC = "sync"
    REPLAY_SEQUENCE = "replay_sequence"
    MULTIPROCESS = "multiprocess"
    ASYNC = "async"
    DISTRIBUTED = "distributed"


@dataclass(frozen=True, slots=True)
class RecurrentStateTransition:
    """One recurrent state key consumed now and produced for the next step."""

    input_key: tuple[str, ...]
    output_key: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecurrentSemantics:
    """Serializable recurrent state, reset, and sequence-window semantics."""

    transitions: tuple[RecurrentStateTransition, ...]
    reset_keys: tuple[tuple[str, ...], ...]
    sequence_dimension: str | None = None
    burn_in: int = 0
    truncated_window: int | None = None
    collector_mode: RecurrentCollectorMode = RecurrentCollectorMode.DIRECT

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("recurrent semantics require at least one state transition")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if self.truncated_window is not None and self.truncated_window < 1:
            raise ValueError("truncated_window must be positive")
        if self.truncated_window is not None and self.burn_in >= self.truncated_window:
            raise ValueError("burn_in must be smaller than truncated_window")
        supported = {
            RecurrentCollectorMode.DIRECT,
            RecurrentCollectorMode.SYNC,
            RecurrentCollectorMode.REPLAY_SEQUENCE,
        }
        if self.collector_mode not in supported:
            raise NotImplementedError(
                f"recurrent collector mode {self.collector_mode.value!r} is unsupported; "
                "only direct, sync, and replay_sequence state lifecycles are validated"
            )


@dataclass(frozen=True, slots=True)
class AgentSelector:
    """A semantic group selection, independent of the implementing module path."""

    group: str
    agents: tuple[str | int, ...] = ()

    def __post_init__(self) -> None:
        if not self.group:
            raise ValueError("agent selector group must be non-empty")
        if len(set(self.agents)) != len(self.agents):
            raise ValueError("agent selector contains duplicate agents")


@dataclass(frozen=True, slots=True)
class SemanticTarget:
    """Target a model role and agent group without treating paths as identities."""

    role: ModelRole
    selector: AgentSelector


@dataclass(frozen=True, slots=True)
class MultiAgentSemantics:
    """Serializable multi-agent topology and semantic target."""

    topology: InteractionTopology
    group: str
    n_agents: int
    target: SemanticTarget

    def __post_init__(self) -> None:
        if not self.group:
            raise ValueError("multi-agent group must be non-empty")
        if self.n_agents < 1:
            raise ValueError("n_agents must be positive")
        if self.target.selector.group != self.group:
            raise ValueError("semantic target group must match the multi-agent group")
        for agent in self.target.selector.agents:
            if isinstance(agent, int) and not 0 <= agent < self.n_agents:
                raise ValueError(f"agent index {agent} is outside the declared group size {self.n_agents}")


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
    model_id: str | None = None
    checkpoint_id: str | None = None
    module_training: bool | None = None
    recurrent: RecurrentSemantics | None = None
    multi_agent: MultiAgentSemantics | None = None

    def __post_init__(self) -> None:
        if self.recurrent is not None:
            _validate_recurrent_descriptor(self)
        if self.multi_agent is not None:
            _validate_multi_agent_descriptor(self)

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
    observations: ObservationTrace | None = None
    interventions: InterventionController | None = None
    events: list[LifecycleEvent] = field(default_factory=list, init=False)
    _stack: ExitStack | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if SchemaSnapshot.from_schema(self.input_schema) != self.descriptor.input_schema:
            raise ValueError("input_schema does not match the interaction descriptor snapshot")
        if SchemaSnapshot.from_schema(self.output_schema) != self.descriptor.output_schema:
            raise ValueError("output_schema does not match the interaction descriptor snapshot")
        self.input_schema.validate_inputs(self.representative_input)
        self._validate_recurrent_input(self.representative_input)
        if self.interventions is not None:
            self.interventions.validate(self)

    def __enter__(self) -> RuntimeInteractionContext:
        if self._stack is not None:
            raise RuntimeError("interaction context is already active")
        stack = ExitStack()
        try:
            if self.descriptor.module_training is not None:
                _set_training_mode(stack, self.module, self.descriptor.module_training)
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

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Run one interaction, making the context usable as a synchronous policy.

        This one-shot form is suitable for direct calls, deterministic rollouts,
        and local :class:`~torchrl.collectors.SyncDataCollector` policies. Keep
        the context open explicitly when hooks must remain installed through a
        subsequent backward pass.
        """
        with self:
            return self.invoke(tensordict)

    def __exit__(self, *exc_info: object) -> bool | None:
        if self._stack is None:
            return None
        stack, self._stack = self._stack, None
        return stack.__exit__(*exc_info)

    def invoke(self, tensordict: TensorDictBase, *, module: TensorDictModuleBase | None = None) -> TensorDictBase:
        """Invoke the wrapped module and append before/after/failure events."""
        if self._stack is None:
            raise RuntimeError("invoke must be called inside the interaction context")
        invoked_module = self.module if module is None else module
        if module is not None and module is not self.module and self.descriptor.module_training is not None:
            _set_training_mode(self._stack, invoked_module, self.descriptor.module_training)
        return self.invoke_callable(tensordict, invoked_module)

    def invoke_callable(
        self,
        tensordict: TensorDictBase,
        operation: Callable[[TensorDictBase], TensorDictBase],
        *,
        module: object | None = None,
    ) -> TensorDictBase:
        """Invoke one TensorDict operation through this interaction's live contract.

        This is the execution boundary used when another library owns the
        model wrapper, as TDHook does for planned method calls.  The context
        must already be active so execution modes and cleanup remain owned by
        the surrounding interaction.
        """
        if self._stack is None:
            raise RuntimeError("invoke_callable must be called inside the interaction context")
        if module is not None and module is not self.module and self.descriptor.module_training is not None:
            _set_training_mode(self._stack, module, self.descriptor.module_training)
        self._record(LifecycleEventType.BEFORE, tensordict)
        try:
            self.input_schema.validate_inputs(tensordict)
            if self.interventions is not None:
                tensordict = self.interventions.apply(self, tensordict, InterventionTiming.INPUT)
            self._validate_recurrent_input(tensordict)
            self._capture_observations(tensordict, input=True)
            result = operation(tensordict)
        except BaseException as error:
            self._record(LifecycleEventType.FAILURE, tensordict, error)
            raise
        try:
            self.output_schema.validate_outputs(result)
            self._validate_recurrent_output(tensordict, result)
            if self.interventions is not None:
                result = self.interventions.apply(self, result, InterventionTiming.OUTPUT)
        except BaseException as error:
            self._record(LifecycleEventType.FAILURE, result, error)
            raise
        self._record(LifecycleEventType.AFTER, result)
        self._capture_observations(result, input=False)
        return result

    def _validate_recurrent_input(self, tensordict: TensorDictBase) -> None:
        recurrent = self.descriptor.recurrent
        if recurrent is None:
            return
        for reset_key in recurrent.reset_keys:
            reset = tensordict.get(reset_key)
            if not isinstance(reset, torch.Tensor) or reset.dtype is not torch.bool:
                raise ValueError(f"recurrent reset key {'/'.join(reset_key)} must contain a boolean tensor")

    def _validate_recurrent_output(self, inputs: TensorDictBase, outputs: TensorDictBase) -> None:
        recurrent = self.descriptor.recurrent
        if recurrent is None:
            return
        for transition in recurrent.transitions:
            previous = inputs.get(transition.input_key)
            following = outputs.get(transition.output_key)
            if not isinstance(previous, torch.Tensor) or not isinstance(following, torch.Tensor):
                raise ValueError("recurrent state transitions must connect tensor-valued keys")
            if previous.shape != following.shape or previous.dtype != following.dtype:
                raise ValueError(
                    f"recurrent state transition {'/'.join(transition.input_key)} -> "
                    f"{'/'.join(transition.output_key)} changed shape or dtype"
                )

    def _capture_observations(self, tensordict: TensorDictBase, *, input: bool) -> None:
        if self.observations is None:
            return
        from xdrl.observations import HookDirection

        schema = self.input_schema if input else self.output_schema
        roles = {_key_path(entry.key): entry.role for entry in schema.keys}
        self.observations.capture_tensordict(
            self.descriptor,
            tensordict,
            direction=HookDirection.INPUT if input else HookDirection.OUTPUT,
            roles=roles,
        )

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


def _validate_recurrent_descriptor(descriptor: InteractionDescriptor) -> None:
    recurrent = descriptor.recurrent
    assert recurrent is not None
    input_entries = {key.path: key for key in descriptor.input_schema.keys}
    output_entries = {key.path: key for key in descriptor.output_schema.keys}
    for transition in recurrent.transitions:
        input_entry = input_entries.get(transition.input_key)
        if input_entry is None:
            raise ValueError(f"recurrent input state key {'/'.join(transition.input_key)} is not declared")
        output_entry = output_entries.get(transition.output_key)
        if output_entry is None:
            raise ValueError(f"recurrent output state key {'/'.join(transition.output_key)} is not declared")
        if input_entry.role != KeyRole.STATE.value or input_entry.presence != KeyPresence.REQUIRED.value:
            raise ValueError(f"recurrent input state key {'/'.join(transition.input_key)} must be required state")
        if output_entry.role != KeyRole.STATE.value or output_entry.presence != KeyPresence.PRODUCED.value:
            raise ValueError(f"recurrent output state key {'/'.join(transition.output_key)} must be produced state")
    for reset_key in recurrent.reset_keys:
        if reset_key not in input_entries:
            raise ValueError(f"recurrent reset key {'/'.join(reset_key)} is not declared")
    if recurrent.sequence_dimension != descriptor.time_dimension:
        raise ValueError("recurrent sequence_dimension must match descriptor time_dimension")
    if recurrent.sequence_dimension is not None and recurrent.sequence_dimension not in descriptor.batch_dimensions:
        raise ValueError("recurrent sequence_dimension must name a declared batch dimension")


def _validate_multi_agent_descriptor(descriptor: InteractionDescriptor) -> None:
    semantics = descriptor.multi_agent
    assert semantics is not None
    if descriptor.agent_dimension is None:
        raise ValueError("multi-agent interactions require an explicit agent_dimension")
    if semantics.target.role is not descriptor.role:
        raise ValueError("semantic target role must match the interaction model role")
    if semantics.topology is InteractionTopology.CENTRALISED_CRITIC and descriptor.role not in {
        ModelRole.CRITIC,
        ModelRole.VALUE,
    }:
        raise ValueError("centralised-critic interactions require a critic or value role")
    if semantics.topology is InteractionTopology.MIXER and descriptor.role is not ModelRole.MIXER:
        raise ValueError("mixer interactions require the mixer model role")


def _restore_training_states(states: tuple[tuple[torch.nn.Module, bool], ...]) -> None:
    """Restore the exact training flag of every submodule in a module tree."""
    for module, training in states:
        module.training = training


def _set_training_mode(stack: ExitStack, module: object, training: bool) -> None:
    """Apply a temporary mode to the module tree actually used for execution."""
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module_training requires the invoked module to be a torch.nn.Module")
    training_states = tuple((child, child.training) for child in module.modules())
    stack.callback(_restore_training_states, training_states)
    module.train(training)


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
