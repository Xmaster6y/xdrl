"""Declarative semantics for typed reinforcement-learning interactions.

These immutable values describe execution meaning without owning models, tensors,
or hook state. Runtime orchestration remains in :mod:`xdrl.interactions`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from xdrl.types import ModelRole, TensorDictKey


__all__ = [
    "AgentIdentity",
    "AgentSelector",
    "CoalitionTerm",
    "InteractionPhase",
    "InteractionTopology",
    "InternalComputationAxis",
    "InternalComputationSemantics",
    "InternalCoordinate",
    "InternalOccurrence",
    "InternalOccurrenceSelection",
    "LifecycleEventType",
    "MultiAgentSemantics",
    "NamedReduction",
    "OccurrenceIdentityError",
    "RecurrentCollectorMode",
    "RecurrentSemantics",
    "RecurrentStateTransition",
    "SemanticTarget",
    "ValueDecompositionAxes",
    "ValueDecompositionKeys",
    "ValueDecompositionSemantics",
]


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


class OccurrenceIdentityError(RuntimeError):
    """Internal hook calls did not match their declared semantic identities."""


InternalCoordinate = str | int


@dataclass(frozen=True, slots=True)
class InternalComputationAxis:
    """One named, architecture-independent internal-computation dimension."""

    name: str
    coordinates: tuple[InternalCoordinate, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("internal-computation axis name must be non-empty")
        if not self.coordinates:
            raise ValueError(f"internal-computation axis {self.name!r} requires coordinates")
        if any(type(value) not in {str, int} or value == "" for value in self.coordinates):
            raise TypeError("internal-computation coordinates must be non-empty strings or integers")
        if len(set(self.coordinates)) != len(self.coordinates):
            raise ValueError(f"internal-computation axis {self.name!r} contains duplicate coordinates")


@dataclass(frozen=True, slots=True)
class InternalOccurrence:
    """Bind semantic coordinates to one raw call of a target module.

    ``call_index`` is zero-based within one root interaction call. It is an
    execution mapping, not an internal-computation dimension.
    """

    module_path: str
    call_index: int
    coordinates: tuple[InternalCoordinate, ...]

    def __post_init__(self) -> None:
        if not self.module_path:
            raise ValueError("internal occurrence module_path must be non-empty")
        if type(self.call_index) is not int or self.call_index < 0:
            raise ValueError("internal occurrence call_index must be a non-negative integer")
        if any(type(value) not in {str, int} or value == "" for value in self.coordinates):
            raise TypeError("internal occurrence coordinates must be non-empty strings or integers")


@dataclass(frozen=True, slots=True)
class InternalOccurrenceSelection:
    """Select internal occurrences by named semantic coordinates."""

    coordinates: tuple[tuple[str, InternalCoordinate], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _value in self.coordinates)
        if any(not name for name in names):
            raise ValueError("internal occurrence selection axis names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("internal occurrence selection contains duplicate axes")
        if any(type(value) not in {str, int} or value == "" for _name, value in self.coordinates):
            raise TypeError("internal occurrence selection values must be non-empty strings or integers")


@dataclass(frozen=True, slots=True)
class InternalComputationSemantics:
    """Serializable semantic axes and their exact per-root-call hook mapping.

    Environment time and sequence/burn-in remain owned by
    :class:`InteractionContract` and :class:`RecurrentSemantics`. These axes
    name only repeated computation inside one root model call. ``occurrences``
    explicitly bridges those semantics to raw hook-call order.
    """

    axes: tuple[InternalComputationAxis, ...]
    occurrences: tuple[InternalOccurrence, ...]
    recurrent_state_keys: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("internal-computation semantics require at least one axis")
        names = tuple(axis.name for axis in self.axes)
        if len(set(names)) != len(names):
            raise ValueError("internal-computation axis names must be unique")
        if not self.occurrences:
            raise ValueError("internal-computation semantics require occurrence mappings")
        if not self.recurrent_state_keys:
            raise ValueError("internal computation must identify related recurrent state keys")
        if len(set(self.recurrent_state_keys)) != len(self.recurrent_state_keys):
            raise ValueError("internal computation contains duplicate recurrent state keys")
        semantic_identities: set[tuple[InternalCoordinate, ...]] = set()
        raw_identities: set[tuple[str, int]] = set()
        indices_by_path: dict[str, set[int]] = {}
        for occurrence in self.occurrences:
            if len(occurrence.coordinates) != len(self.axes):
                raise ValueError("internal occurrence must provide one coordinate for every declared axis")
            for axis, coordinate in zip(self.axes, occurrence.coordinates):
                if coordinate not in axis.coordinates:
                    raise ValueError(
                        f"coordinate {coordinate!r} is not declared on internal-computation axis {axis.name!r}"
                    )
            if occurrence.coordinates in semantic_identities:
                raise ValueError("internal occurrence semantic coordinates must be unique")
            raw_identity = (occurrence.module_path, occurrence.call_index)
            if raw_identity in raw_identities:
                raise ValueError("one raw hook call cannot identify multiple internal occurrences")
            semantic_identities.add(occurrence.coordinates)
            raw_identities.add(raw_identity)
            indices_by_path.setdefault(occurrence.module_path, set()).add(occurrence.call_index)
        for module_path, indices in indices_by_path.items():
            if indices != set(range(len(indices))):
                raise ValueError(f"internal occurrence call indices for {module_path!r} must be contiguous from zero")

    def select(self, selection: InternalOccurrenceSelection) -> tuple[InternalOccurrence, ...]:
        """Resolve a semantic selection to exact raw occurrences or fail."""
        axis_names = tuple(axis.name for axis in self.axes)
        requested = dict(selection.coordinates)
        unknown = set(requested) - set(axis_names)
        if unknown:
            raise OccurrenceIdentityError(f"unknown internal-computation axes: {', '.join(sorted(unknown))}")
        for axis in self.axes:
            if axis.name in requested and requested[axis.name] not in axis.coordinates:
                raise OccurrenceIdentityError(
                    f"coordinate {requested[axis.name]!r} is not declared on axis {axis.name!r}"
                )
        matches = tuple(
            occurrence
            for occurrence in self.occurrences
            if all(
                axis.name not in requested or requested[axis.name] == coordinate
                for axis, coordinate in zip(self.axes, occurrence.coordinates)
            )
        )
        if not matches:
            raise OccurrenceIdentityError("internal occurrence selection matched no declared calls")
        return matches

    def coordinates_for(self, occurrence: InternalOccurrence) -> tuple[tuple[str, InternalCoordinate], ...]:
        """Return named coordinates for one occurrence in canonical axis order."""
        return tuple((axis.name, value) for axis, value in zip(self.axes, occurrence.coordinates))


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
        if type(self.burn_in) is not int or self.burn_in < 0:
            raise ValueError("burn_in must be a non-negative integer")
        if self.truncated_window is not None and (type(self.truncated_window) is not int or self.truncated_window < 1):
            raise ValueError("truncated_window must be a positive integer")
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


AgentIdentity = str | int


@dataclass(frozen=True, slots=True)
class CoalitionTerm:
    """One semantic coalition coordinate, independent of module structure."""

    identity: str
    members: tuple[AgentIdentity, ...]
    axis_index: int

    def __post_init__(self) -> None:
        if type(self.identity) is not str:
            raise TypeError("coalition identity must be a string")
        if not self.identity:
            raise ValueError("coalition identity must be non-empty")
        if not self.members:
            raise ValueError("coalition membership must be non-empty")
        if any(type(member) not in {str, int} or member == "" for member in self.members):
            raise TypeError("coalition members must be non-empty strings or integers")
        if len(set(self.members)) != len(self.members):
            raise ValueError("coalition membership contains duplicate agents")
        if type(self.axis_index) is not int or self.axis_index < 0:
            raise ValueError("coalition axis_index must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ValueDecompositionKeys:
    """Nested TensorDict keys used by a value-decomposition interaction."""

    individual_value: TensorDictKey
    coalition_contribution: TensorDictKey
    semantic_mask: TensorDictKey
    mixer_input: TensorDictKey
    joint_value: TensorDictKey

    def __post_init__(self) -> None:
        paths = tuple(_key_path(key) for key in asdict(self).values())
        if any(not path or any(not part for part in path) for path in paths):
            raise ValueError("value-decomposition keys must be non-empty")
        if len(set(paths)) != len(paths):
            raise ValueError("value-decomposition keys must be unique")


@dataclass(frozen=True, slots=True)
class ValueDecompositionAxes:
    """Semantic axes for each value-decomposition tensor."""

    individual_value: tuple[str, ...]
    coalition_contribution: tuple[str, ...]
    semantic_mask: tuple[str, ...]
    mixer_input: tuple[str, ...]
    joint_value: tuple[str, ...]

    def __post_init__(self) -> None:
        for role, axes in asdict(self).items():
            if not axes:
                raise ValueError(f"{role} axes must be explicit")
            if any(not axis for axis in axes):
                raise ValueError(f"{role} axes must be non-empty")
            if len(set(axes)) != len(axes):
                raise ValueError(f"{role} axes contain duplicates")


@dataclass(frozen=True, slots=True)
class NamedReduction:
    """A provenance-bearing aggregation between declared TensorDict keys."""

    name: str
    source_key: TensorDictKey
    target_key: TensorDictKey
    reduced_axes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("reduction name must be non-empty")
        if not self.reduced_axes or any(not axis for axis in self.reduced_axes):
            raise ValueError("reduced_axes must be explicit and non-empty")
        if len(set(self.reduced_axes)) != len(self.reduced_axes):
            raise ValueError("reduced_axes contains duplicates")
        if _key_path(self.source_key) == _key_path(self.target_key):
            raise ValueError("reduction source and target keys must differ")


@dataclass(frozen=True, slots=True)
class ValueDecompositionSemantics:
    """Coalition identities, tensor axes, keys, and explicit aggregations."""

    coalition_axis: str
    feature_axes: tuple[str, ...]
    terms: tuple[CoalitionTerm, ...]
    keys: ValueDecompositionKeys
    axes: ValueDecompositionAxes
    reductions: tuple[NamedReduction, ...]
    parameters_shared: bool = False
    coalition_targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.coalition_axis:
            raise ValueError("coalition_axis must be non-empty")
        if any(not axis for axis in self.feature_axes) or len(set(self.feature_axes)) != len(self.feature_axes):
            raise ValueError("feature axes must be non-empty and unique")
        if not self.terms:
            raise ValueError("value decomposition requires coalition terms")
        identities = {term.identity for term in self.terms}
        if len(identities) != len(self.terms):
            raise ValueError("coalition identities must be unique")
        memberships = tuple(term.members for term in self.terms)
        if len(set(memberships)) != len(memberships):
            raise ValueError("coalition memberships must be unique")
        if tuple(term.axis_index for term in self.terms) != tuple(range(len(self.terms))):
            raise ValueError("coalition axis indices must be contiguous and order-stable from zero")
        if not self.reductions:
            raise ValueError("value decomposition requires at least one named reduction")
        if len({reduction.name for reduction in self.reductions}) != len(self.reductions):
            raise ValueError("reduction names must be unique")
        if type(self.parameters_shared) is not bool:
            raise TypeError("parameters_shared must be a boolean")
        if len(set(self.coalition_targets)) != len(self.coalition_targets):
            raise ValueError("coalition targets must be unique")
        unknown_targets = set(self.coalition_targets) - identities
        if unknown_targets:
            raise ValueError(f"unknown coalition targets: {', '.join(sorted(unknown_targets))}")

    @property
    def targeted_terms(self) -> tuple[CoalitionTerm, ...]:
        """Return selected semantic terms in canonical coalition-axis order."""
        if not self.coalition_targets:
            return self.terms
        selected = set(self.coalition_targets)
        return tuple(term for term in self.terms if term.identity in selected)


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
    agent_identities: tuple[AgentIdentity, ...] = ()

    def __post_init__(self) -> None:
        if not self.group:
            raise ValueError("multi-agent group must be non-empty")
        if type(self.n_agents) is not int or self.n_agents < 1:
            raise ValueError("n_agents must be a positive integer")
        if self.target.selector.group != self.group:
            raise ValueError("semantic target group must match the multi-agent group")
        if self.agent_identities:
            if len(self.agent_identities) != self.n_agents:
                raise ValueError("agent_identities must match the declared group size")
            if len(set(self.agent_identities)) != len(self.agent_identities):
                raise ValueError("agent_identities must be unique")
            if any(type(agent) not in {str, int} or agent == "" for agent in self.agent_identities):
                raise TypeError("agent identities must be non-empty strings or integers")
        if self.agent_identities:
            for agent in self.target.selector.agents:
                if not any(type(agent) is type(identity) and agent == identity for identity in self.agent_identities):
                    raise ValueError(f"agent identity {agent!r} is outside the declared agent group")
        else:
            for agent in self.target.selector.agents:
                if type(agent) is not int or not 0 <= agent < self.n_agents:
                    raise ValueError(f"agent index {agent} is outside the declared group size {self.n_agents}")

    @property
    def declared_agents(self) -> tuple[AgentIdentity, ...]:
        """Return the canonical ordered semantic identities for the group."""
        return self.agent_identities or tuple(range(self.n_agents))


def _key_path(key: TensorDictKey) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else tuple(key)
