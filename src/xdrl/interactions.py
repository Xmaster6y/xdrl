"""Typed contracts and execution contexts for TorchRL model invocations.

An :class:`InteractionContract` is the single immutable declaration of the
live TensorDict schemas and RL execution semantics. Its serialised projection
is tensor-free; the runtime context owns only the module, representative batch,
and temporary execution state needed to invoke that module safely.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterator, Protocol

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
class _KeySnapshot:
    """Serialisable description of one declared TensorDict key."""

    path: tuple[str, ...]
    role: str
    presence: str
    feature_shape: tuple[int, ...] | None
    spec_type: str | None
    spec_constraints: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class _SchemaSnapshot:
    """Serialisable projection of a :class:`TensorDictSchema`."""

    keys: tuple[_KeySnapshot, ...]
    batch_dimensions: tuple[str, ...]

    @classmethod
    def from_schema(cls, schema: TensorDictSchema) -> _SchemaSnapshot:
        return cls(
            keys=tuple(
                _KeySnapshot(
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
class InteractionContract:
    """Canonical immutable schemas and semantics for one model interaction.

    ``identity`` must be stable within one recorded run.  Events for a given
    identity are ordered by their monotonically increasing ``order`` field.
    """

    identity: str
    role: ModelRole
    phase: InteractionPhase
    module_path: str
    input_schema: TensorDictSchema
    output_schema: TensorDictSchema
    environment: str | None = None
    time_dimension: str | None = None
    agent_dimension: str | None = None
    objective: str | None = None
    exploration_mode: str | None = None
    gradient_enabled: bool = False
    inference_mode: bool = False
    autocast_device_type: str | None = None
    autocast_enabled: bool = False
    logical_step: int | None = None
    episode_id: str | int | None = None
    trajectory_id: str | int | None = None
    model_id: str | None = None
    checkpoint_id: str | None = None
    module_training: bool | None = None
    recurrent: RecurrentSemantics | None = None
    multi_agent: MultiAgentSemantics | None = None

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("interaction identity must be non-empty")
        if not self.module_path:
            raise ValueError("interaction module_path must be non-empty")
        if self.inference_mode and self.gradient_enabled:
            raise ValueError("inference_mode and gradient_enabled cannot both be enabled")
        if self.autocast_enabled and self.autocast_device_type is None:
            raise ValueError("autocast_enabled requires autocast_device_type")
        if self.input_schema.batch != self.output_schema.batch:
            raise ValueError("input and output schemas must declare identical batch semantics")
        if self.recurrent is not None:
            _validate_recurrent_contract(self)
        if self.multi_agent is not None:
            _validate_multi_agent_contract(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible tensor-free projection of this contract."""
        return {
            "identity": self.identity,
            "role": self.role.value,
            "phase": self.phase.value,
            "module_path": self.module_path,
            "input_schema": asdict(_SchemaSnapshot.from_schema(self.input_schema)),
            "output_schema": asdict(_SchemaSnapshot.from_schema(self.output_schema)),
            "batch_dimensions": self.batch_dimensions,
            "environment": self.environment,
            "time_dimension": self.time_dimension,
            "agent_dimension": self.agent_dimension,
            "objective": self.objective,
            "exploration_mode": self.exploration_mode,
            "gradient_enabled": self.gradient_enabled,
            "inference_mode": self.inference_mode,
            "autocast_device_type": self.autocast_device_type,
            "autocast_enabled": self.autocast_enabled,
            "logical_step": self.logical_step,
            "episode_id": self.episode_id,
            "trajectory_id": self.trajectory_id,
            "model_id": self.model_id,
            "checkpoint_id": self.checkpoint_id,
            "module_training": self.module_training,
            "recurrent": asdict(self.recurrent) if self.recurrent is not None else None,
            "multi_agent": asdict(self.multi_agent) if self.multi_agent is not None else None,
        }

    @property
    def batch_dimensions(self) -> tuple[str, ...]:
        """Return the single batch declaration shared by both schemas."""
        return self.input_schema.batch.dimensions


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_shapes", MappingProxyType(dict(self.key_shapes)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible event representation."""
        return {
            "order": self.order,
            "kind": self.kind.value,
            "interaction_id": self.interaction_id,
            "phase": self.phase.value,
            "module_path": self.module_path,
            "key_shapes": dict(self.key_shapes),
            "error": self.error,
        }


class HookContextFactory(Protocol):
    """Create a context manager that installs temporary TDHook state."""

    def __call__(self) -> Any: ...


@dataclass(slots=True)
class RuntimeInteractionContext:
    """Ephemeral execution wrapper around one existing TensorDict module.

    Construction validates the supplied representative input.  ``invoke``
    validates the live input/output around the actual module call and records
    lifecycle metadata.  The module itself, tensors, and hook state never
    enter :class:`InteractionContract` or :class:`LifecycleEvent`.
    """

    contract: InteractionContract
    module: TensorDictModuleBase
    representative_input: TensorDictBase
    hook_context_factory: HookContextFactory | None = None
    observations: ObservationTrace | None = None
    interventions: InterventionController | None = None
    events: list[LifecycleEvent] = field(default_factory=list, init=False)
    _stack: ExitStack | None = field(default=None, init=False, repr=False)
    _observing_module_calls: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.input_schema.validate_inputs(self.representative_input)
        self._validate_recurrent_input(self.representative_input)
        if self.interventions is not None:
            self.interventions.validate(self)

    @property
    def input_schema(self) -> TensorDictSchema:
        """Return the contract's canonical input schema."""
        return self.contract.input_schema

    @property
    def output_schema(self) -> TensorDictSchema:
        """Return the contract's canonical output schema."""
        return self.contract.output_schema

    def __enter__(self) -> RuntimeInteractionContext:
        if self._stack is not None:
            raise RuntimeError("interaction context is already active")
        stack = ExitStack()
        try:
            if self.contract.module_training is not None:
                _set_training_mode(stack, self.module, self.contract.module_training)
            if self.contract.exploration_mode is not None:
                stack.enter_context(set_exploration_type(self.contract.exploration_mode))
            stack.enter_context(torch.inference_mode(self.contract.inference_mode))
            if not self.contract.inference_mode:
                stack.enter_context(torch.set_grad_enabled(self.contract.gradient_enabled))
            if self.contract.autocast_device_type is not None:
                stack.enter_context(
                    torch.autocast(
                        device_type=self.contract.autocast_device_type,
                        enabled=self.contract.autocast_enabled,
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
        if module is not None and module is not self.module and self.contract.module_training is not None:
            _set_training_mode(self._stack, invoked_module, self.contract.module_training)
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
        if module is not None and module is not self.module and self.contract.module_training is not None:
            _set_training_mode(self._stack, module, self.contract.module_training)
        current = self._before_call(tensordict)
        try:
            result = operation(current)
        except BaseException as error:
            self._record(LifecycleEventType.FAILURE, current, error)
            raise
        return self._after_call(current, result)

    @contextmanager
    def observe_module_calls(self) -> Iterator[None]:
        """Validate every root-module call made by another execution owner.

        TDHook workflows must receive the original module so their model-relative
        targets and binding facts remain unchanged. Temporary public root hooks
        let XDRL observe each actual model pass without wrapping the module or
        changing its class.
        """
        if self._stack is None:
            raise RuntimeError("observe_module_calls requires an active interaction context")
        if self._observing_module_calls:
            raise RuntimeError("module-call observation is already active")
        pending: list[TensorDictBase] = []

        def before(_module: torch.nn.Module, args: tuple[object, ...]) -> tuple[object, ...]:
            if _module is not self.module:
                return args
            if len(args) != 1 or not isinstance(args[0], TensorDictBase):
                raise TypeError("observed model calls require one positional TensorDict argument")
            current = self._before_call(args[0])
            pending.append(current)
            return (current,)

        def after(_module: torch.nn.Module, _args: tuple[object, ...], result: object) -> TensorDictBase:
            if _module is not self.module:
                if not isinstance(result, TensorDictBase):
                    raise TypeError(f"TDHook wrapper must return a TensorDict, got {type(result).__name__}")
                return result
            if not pending:
                raise RuntimeError("observed model call completed without a matching input")
            current = pending.pop()
            if not isinstance(result, TensorDictBase):
                error = TypeError(f"observed model call must return a TensorDict, got {type(result).__name__}")
                self._record(LifecycleEventType.FAILURE, current, error)
                raise error
            return self._after_call(current, result)

        self._observing_module_calls = True
        pre_handle = self.module.register_forward_pre_hook(before, prepend=True)
        post_handle = self.module.register_forward_hook(after)
        try:
            yield
        except BaseException as error:
            while pending:
                self._record(LifecycleEventType.FAILURE, pending.pop(), error)
            raise
        finally:
            post_handle.remove()
            pre_handle.remove()
            self._observing_module_calls = False

    def _before_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._record(LifecycleEventType.BEFORE, tensordict)
        try:
            self.input_schema.validate_inputs(tensordict)
            if self.interventions is not None:
                tensordict = self.interventions.apply(self, tensordict, InterventionTiming.INPUT)
            self._validate_recurrent_input(tensordict)
            self._capture_observations(tensordict, input=True)
        except BaseException as error:
            self._record(LifecycleEventType.FAILURE, tensordict, error)
            raise
        return tensordict

    def _after_call(self, inputs: TensorDictBase, result: TensorDictBase) -> TensorDictBase:
        try:
            self.output_schema.validate_outputs(result)
            self._validate_recurrent_output(inputs, result)
            if self.interventions is not None:
                result = self.interventions.apply(self, result, InterventionTiming.OUTPUT)
        except BaseException as error:
            self._record(LifecycleEventType.FAILURE, result, error)
            raise
        self._record(LifecycleEventType.AFTER, result)
        self._capture_observations(result, input=False)
        return result

    def _validate_recurrent_input(self, tensordict: TensorDictBase) -> None:
        recurrent = self.contract.recurrent
        if recurrent is None:
            return
        for reset_key in recurrent.reset_keys:
            reset = tensordict.get(reset_key)
            if not isinstance(reset, torch.Tensor) or reset.dtype is not torch.bool:
                raise ValueError(f"recurrent reset key {'/'.join(reset_key)} must contain a boolean tensor")

    def _validate_recurrent_output(self, inputs: TensorDictBase, outputs: TensorDictBase) -> None:
        recurrent = self.contract.recurrent
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
            self.contract,
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
                interaction_id=self.contract.identity,
                phase=self.contract.phase,
                module_path=self.contract.module_path,
                key_shapes=_key_shapes(tensordict),
                error=f"{type(error).__name__}: {error}" if error is not None else None,
            )
        )


def _key_path(key: TensorDictKey) -> tuple[str, ...]:
    return tuple(str(part) for part in key) if isinstance(key, tuple) else (str(key),)


def _validate_recurrent_contract(contract: InteractionContract) -> None:
    recurrent = contract.recurrent
    assert recurrent is not None
    input_entries = {_key_path(entry.key): entry for entry in contract.input_schema.keys}
    output_entries = {_key_path(entry.key): entry for entry in contract.output_schema.keys}
    for transition in recurrent.transitions:
        input_entry = input_entries.get(transition.input_key)
        if input_entry is None:
            raise ValueError(f"recurrent input state key {'/'.join(transition.input_key)} is not declared")
        output_entry = output_entries.get(transition.output_key)
        if output_entry is None:
            raise ValueError(f"recurrent output state key {'/'.join(transition.output_key)} is not declared")
        if input_entry.role is not KeyRole.STATE or input_entry.presence is not KeyPresence.REQUIRED:
            raise ValueError(f"recurrent input state key {'/'.join(transition.input_key)} must be required state")
        if output_entry.role is not KeyRole.STATE or output_entry.presence is not KeyPresence.PRODUCED:
            raise ValueError(f"recurrent output state key {'/'.join(transition.output_key)} must be produced state")
    for reset_key in recurrent.reset_keys:
        if reset_key not in input_entries:
            raise ValueError(f"recurrent reset key {'/'.join(reset_key)} is not declared")
    if recurrent.sequence_dimension != contract.time_dimension:
        raise ValueError("recurrent sequence_dimension must match contract time_dimension")
    if recurrent.sequence_dimension is not None and recurrent.sequence_dimension not in contract.batch_dimensions:
        raise ValueError("recurrent sequence_dimension must name a declared batch dimension")


def _validate_multi_agent_contract(contract: InteractionContract) -> None:
    semantics = contract.multi_agent
    assert semantics is not None
    if contract.agent_dimension is None:
        raise ValueError("multi-agent interactions require an explicit agent_dimension")
    if semantics.target.role is not contract.role:
        raise ValueError("semantic target role must match the interaction model role")
    if semantics.topology is InteractionTopology.CENTRALISED_CRITIC and contract.role not in {
        ModelRole.CRITIC,
        ModelRole.VALUE,
    }:
        raise ValueError("centralised-critic interactions require a critic or value role")
    if semantics.topology is InteractionTopology.MIXER and contract.role is not ModelRole.MIXER:
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
