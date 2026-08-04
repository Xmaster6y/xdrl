import json

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import UnboundedContinuous
from torchrl.modules import LSTMModule, MultiAgentMLP, QMixer

from xdrl.interactions import (
    AgentSelector,
    InteractionDescriptor,
    InteractionPhase,
    InteractionTopology,
    MultiAgentSemantics,
    RecurrentCollectorMode,
    RecurrentSemantics,
    RecurrentStateTransition,
    RuntimeInteractionContext,
    SchemaSnapshot,
    SemanticTarget,
)
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


def _descriptor(
    role: ModelRole,
    module_path: str,
    inputs: TensorDictSchema,
    outputs: TensorDictSchema,
    **kwargs: object,
) -> InteractionDescriptor:
    return InteractionDescriptor(
        identity=f"{role.value}:0",
        role=role,
        phase=InteractionPhase.COLLECTION,
        module_path=module_path,
        input_schema=SchemaSnapshot.from_schema(inputs),
        output_schema=SchemaSnapshot.from_schema(outputs),
        batch_dimensions=inputs.batch.dimensions,
        **kwargs,
    )


def _state_schemas(dimensions: tuple[str, ...] = ("env",)) -> tuple[TensorDictSchema, TensorDictSchema]:
    return (
        TensorDictSchema(
            (
                KeySchema("state", KeyRole.STATE, KeyPresence.REQUIRED),
                KeySchema("is_init", KeyRole.TERMINATION, KeyPresence.REQUIRED),
            ),
            BatchSemantics(dimensions),
        ),
        TensorDictSchema(
            (KeySchema(("next", "state"), KeyRole.STATE, KeyPresence.PRODUCED),),
            BatchSemantics(dimensions),
        ),
    )


def test_torchrl_lstm_state_transitions_and_reset_masks_are_validated() -> None:
    inputs = TensorDictSchema(
        (
            KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),
            KeySchema("rs_h", KeyRole.STATE, KeyPresence.REQUIRED),
            KeySchema("rs_c", KeyRole.STATE, KeyPresence.REQUIRED),
            KeySchema("is_init", KeyRole.TERMINATION, KeyPresence.REQUIRED),
        ),
        BatchSemantics(("env",)),
    )
    outputs = TensorDictSchema(
        (
            KeySchema("intermediate", KeyRole.FEATURE, KeyPresence.PRODUCED),
            KeySchema(("next", "rs_h"), KeyRole.STATE, KeyPresence.PRODUCED),
            KeySchema(("next", "rs_c"), KeyRole.STATE, KeyPresence.PRODUCED),
        ),
        BatchSemantics(("env",)),
    )
    recurrent = RecurrentSemantics(
        transitions=(
            RecurrentStateTransition(("rs_h",), ("next", "rs_h")),
            RecurrentStateTransition(("rs_c",), ("next", "rs_c")),
        ),
        reset_keys=(("is_init",),),
        collector_mode=RecurrentCollectorMode.SYNC,
    )
    descriptor = _descriptor(ModelRole.ACTOR, "policy.lstm", inputs, outputs, recurrent=recurrent)
    module = LSTMModule(
        input_size=3,
        hidden_size=4,
        in_keys=["observation", "rs_h", "rs_c"],
        out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")],
    )
    state = torch.ones(2, 1, 4)
    batch = TensorDict(
        {
            "observation": torch.randn(2, 3),
            "rs_h": state.clone(),
            "rs_c": state.clone(),
            "is_init": torch.tensor([[True], [False]]),
        },
        batch_size=[2],
    )
    zero_state = batch.clone().set("rs_h", torch.zeros_like(state)).set("rs_c", torch.zeros_like(state))
    expected_reset = module(zero_state)["next", "rs_h"][0].clone()
    context = RuntimeInteractionContext(descriptor, module, inputs, outputs, batch)

    result = context(batch.clone())

    assert torch.allclose(result["next", "rs_h"][0], expected_reset)
    assert result["next", "rs_h"].shape == state.shape
    json.dumps(descriptor.to_dict())


def test_recurrent_contract_rejects_bad_masks_and_unsupported_collectors() -> None:
    with pytest.raises(NotImplementedError, match="multiprocess"):
        RecurrentSemantics(
            transitions=(RecurrentStateTransition(("state",), ("next", "state")),),
            reset_keys=(("is_init",),),
            collector_mode=RecurrentCollectorMode.MULTIPROCESS,
        )

    inputs = TensorDictSchema(
        (
            KeySchema("state", KeyRole.STATE, KeyPresence.REQUIRED),
            KeySchema("is_init", KeyRole.TERMINATION, KeyPresence.REQUIRED),
        ),
        BatchSemantics(("env",)),
    )
    outputs = TensorDictSchema(
        (KeySchema(("next", "state"), KeyRole.STATE, KeyPresence.PRODUCED),), BatchSemantics(("env",))
    )
    recurrent = RecurrentSemantics(
        transitions=(RecurrentStateTransition(("state",), ("next", "state")),),
        reset_keys=(("is_init",),),
    )
    descriptor = _descriptor(ModelRole.ACTOR, "policy.rnn", inputs, outputs, recurrent=recurrent)
    batch = TensorDict(
        {"state": torch.zeros(2, 1), "is_init": torch.zeros(2, 1, dtype=torch.bool)},
        batch_size=[2],
    )
    context = RuntimeInteractionContext(descriptor, torch.nn.Identity(), inputs, outputs, batch)

    with pytest.raises(ValueError, match="boolean tensor"):
        with context:
            context.invoke(batch.clone().set("is_init", torch.zeros(2, 1)))


@pytest.mark.parametrize(
    ("transitions", "burn_in", "truncated_window", "message"),
    [
        ((), 0, None, "at least one"),
        ((RecurrentStateTransition(("state",), ("next", "state")),), -1, None, "non-negative"),
        ((RecurrentStateTransition(("state",), ("next", "state")),), 0, 0, "positive"),
        ((RecurrentStateTransition(("state",), ("next", "state")),), 2, 2, "smaller"),
    ],
)
def test_recurrent_semantics_reject_invalid_windows(
    transitions: tuple[RecurrentStateTransition, ...],
    burn_in: int,
    truncated_window: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RecurrentSemantics(transitions, (), burn_in=burn_in, truncated_window=truncated_window)


def test_agent_and_multi_agent_semantics_reject_invalid_selectors() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AgentSelector("")
    with pytest.raises(ValueError, match="duplicate"):
        AgentSelector("agents", (1, 1))
    selector = AgentSelector("agents")

    with pytest.raises(ValueError, match="non-empty"):
        MultiAgentSemantics(InteractionTopology.INDEPENDENT, "", 2, SemanticTarget(ModelRole.ACTOR, selector))
    with pytest.raises(ValueError, match="positive"):
        MultiAgentSemantics(InteractionTopology.INDEPENDENT, "agents", 0, SemanticTarget(ModelRole.ACTOR, selector))
    with pytest.raises(ValueError, match="must match"):
        MultiAgentSemantics(
            InteractionTopology.INDEPENDENT,
            "agents",
            2,
            SemanticTarget(ModelRole.ACTOR, AgentSelector("other")),
        )
    with pytest.raises(ValueError, match="outside"):
        MultiAgentSemantics(
            InteractionTopology.INDEPENDENT,
            "agents",
            2,
            SemanticTarget(ModelRole.ACTOR, AgentSelector("agents", (2,))),
        )


def test_recurrent_contract_rejects_non_state_transition_keys() -> None:
    inputs = TensorDictSchema(
        (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),), BatchSemantics(("env",))
    )
    outputs = TensorDictSchema((KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),), BatchSemantics(("env",)))
    recurrent = RecurrentSemantics(
        transitions=(RecurrentStateTransition(("observation",), ("action",)),),
        reset_keys=(),
    )

    with pytest.raises(ValueError, match="must be required state"):
        _descriptor(ModelRole.ACTOR, "policy.rnn", inputs, outputs, recurrent=recurrent)


@pytest.mark.parametrize(
    ("transition", "reset_keys", "time_dimension", "sequence_dimension", "message"),
    [
        (RecurrentStateTransition(("missing",), ("next", "state")), (), None, None, "input state key.*not declared"),
        (RecurrentStateTransition(("state",), ("next", "missing")), (), None, None, "output state key.*not declared"),
        (
            RecurrentStateTransition(("state",), ("next", "state")),
            (("missing",),),
            None,
            None,
            "reset key.*not declared",
        ),
        (RecurrentStateTransition(("state",), ("next", "state")), (), "other", "time", "must match"),
        (RecurrentStateTransition(("state",), ("next", "state")), (), "time", "time", "declared batch"),
    ],
)
def test_recurrent_descriptor_rejects_invalid_key_and_time_declarations(
    transition: RecurrentStateTransition,
    reset_keys: tuple[tuple[str, ...], ...],
    time_dimension: str | None,
    sequence_dimension: str | None,
    message: str,
) -> None:
    inputs, outputs = _state_schemas()
    recurrent = RecurrentSemantics((transition,), reset_keys, sequence_dimension=sequence_dimension)

    with pytest.raises(ValueError, match=message):
        _descriptor(
            ModelRole.ACTOR,
            "policy.rnn",
            inputs,
            outputs,
            time_dimension=time_dimension,
            recurrent=recurrent,
        )


@pytest.mark.parametrize(
    ("topology", "role", "target_role", "agent_dimension", "message"),
    [
        (InteractionTopology.INDEPENDENT, ModelRole.ACTOR, ModelRole.ACTOR, None, "agent_dimension"),
        (InteractionTopology.INDEPENDENT, ModelRole.ACTOR, ModelRole.CRITIC, "agent", "must match"),
        (InteractionTopology.CENTRALISED_CRITIC, ModelRole.ACTOR, ModelRole.ACTOR, "agent", "critic or value"),
        (InteractionTopology.MIXER, ModelRole.ACTOR, ModelRole.ACTOR, "agent", "mixer model role"),
    ],
)
def test_multi_agent_descriptor_rejects_invalid_role_and_axis_contracts(
    topology: InteractionTopology,
    role: ModelRole,
    target_role: ModelRole,
    agent_dimension: str | None,
    message: str,
) -> None:
    schema = TensorDictSchema((), BatchSemantics(("env",)))
    semantics = MultiAgentSemantics(topology, "agents", 2, SemanticTarget(target_role, AgentSelector("agents")))

    with pytest.raises(ValueError, match=message):
        _descriptor(
            role,
            "policy.module",
            schema,
            schema,
            agent_dimension=agent_dimension,
            multi_agent=semantics,
        )


def test_replay_sequence_serialises_time_axis_burn_in_and_truncated_window() -> None:
    inputs = TensorDictSchema(
        (
            KeySchema("state", KeyRole.STATE, KeyPresence.REQUIRED),
            KeySchema("is_init", KeyRole.TERMINATION, KeyPresence.REQUIRED),
        ),
        BatchSemantics(("env", "time")),
    )
    outputs = TensorDictSchema(
        (KeySchema(("next", "state"), KeyRole.STATE, KeyPresence.PRODUCED),),
        BatchSemantics(("env", "time")),
    )
    recurrent = RecurrentSemantics(
        transitions=(RecurrentStateTransition(("state",), ("next", "state")),),
        reset_keys=(("is_init",),),
        sequence_dimension="time",
        burn_in=2,
        truncated_window=8,
        collector_mode=RecurrentCollectorMode.REPLAY_SEQUENCE,
    )
    descriptor = _descriptor(
        ModelRole.ACTOR,
        "policy.rnn",
        inputs,
        outputs,
        time_dimension="time",
        recurrent=recurrent,
    )

    encoded = json.loads(json.dumps(descriptor.to_dict()))["recurrent"]

    assert encoded["sequence_dimension"] == "time"
    assert encoded["burn_in"] == 2
    assert encoded["truncated_window"] == 8


def test_vmas_parameter_sharing_uses_a_semantic_group_target() -> None:
    n_agents = 3
    inputs = TensorDictSchema(
        (
            KeySchema(
                ("agents", "observation"),
                KeyRole.OBSERVATION,
                KeyPresence.REQUIRED,
                UnboundedContinuous(shape=(n_agents, 4)),
            ),
        ),
        BatchSemantics(("env",)),
    )
    outputs = TensorDictSchema(
        (
            KeySchema(
                ("agents", "action"),
                KeyRole.ACTION,
                KeyPresence.PRODUCED,
                UnboundedContinuous(shape=(n_agents, 2)),
            ),
        ),
        BatchSemantics(("env",)),
    )
    semantics = MultiAgentSemantics(
        topology=InteractionTopology.PARAMETER_SHARED,
        group="agents",
        n_agents=n_agents,
        target=SemanticTarget(ModelRole.ACTOR, AgentSelector("agents")),
    )
    descriptor = _descriptor(
        ModelRole.ACTOR,
        "policy.module",
        inputs,
        outputs,
        agent_dimension="agent",
        multi_agent=semantics,
    )
    policy = TensorDictModule(
        MultiAgentMLP(4, 2, n_agents, centralized=False, share_params=True, depth=1, num_cells=8),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "action")],
    )
    batch = TensorDict(
        {("agents", "observation"): torch.randn(5, n_agents, 4)},
        batch_size=[5],
    )

    result = RuntimeInteractionContext(descriptor, policy, inputs, outputs, batch)(batch.clone())

    assert result["agents", "action"].shape == (5, n_agents, 2)
    assert descriptor.multi_agent.target.selector.agents == ()
    assert descriptor.module_path == "policy.module"


@pytest.mark.parametrize(
    ("topology", "role"),
    [
        (InteractionTopology.CENTRALISED_CRITIC, ModelRole.CRITIC),
        (InteractionTopology.MIXER, ModelRole.MIXER),
    ],
)
def test_centralised_critic_and_mixer_roles_are_serialisable(topology: InteractionTopology, role: ModelRole) -> None:
    schema = TensorDictSchema((), BatchSemantics(("env",)))
    semantics = MultiAgentSemantics(topology, "agents", 3, SemanticTarget(role, AgentSelector("agents")))
    descriptor = _descriptor(
        role,
        "loss.mixer" if role is ModelRole.MIXER else "loss.critic",
        schema,
        schema,
        agent_dimension="agent",
        multi_agent=semantics,
    )

    assert json.loads(json.dumps(descriptor.to_dict()))["multi_agent"]["topology"] == topology.value


def test_qmix_reference_preserves_nested_agent_keys_and_environment_batch() -> None:
    n_agents = 3
    inputs = TensorDictSchema(
        (
            KeySchema(("agents", "chosen_action_value"), KeyRole.VALUE, KeyPresence.REQUIRED),
            KeySchema(("agents", "observation"), KeyRole.OBSERVATION, KeyPresence.REQUIRED),
        ),
        BatchSemantics(("env",)),
    )
    outputs = TensorDictSchema(
        (KeySchema("chosen_action_value", KeyRole.VALUE, KeyPresence.PRODUCED),), BatchSemantics(("env",))
    )
    semantics = MultiAgentSemantics(
        InteractionTopology.MIXER,
        "agents",
        n_agents,
        SemanticTarget(ModelRole.MIXER, AgentSelector("agents")),
    )
    descriptor = _descriptor(
        ModelRole.MIXER,
        "loss.mixer_network",
        inputs,
        outputs,
        agent_dimension="agent",
        multi_agent=semantics,
    )
    mixer = TensorDictModule(
        QMixer(state_shape=(n_agents, 4), mixing_embed_dim=8, n_agents=n_agents, device="cpu"),
        in_keys=[("agents", "chosen_action_value"), ("agents", "observation")],
        out_keys=["chosen_action_value"],
    )
    batch = TensorDict(
        {
            ("agents", "chosen_action_value"): torch.randn(2, n_agents, 1),
            ("agents", "observation"): torch.randn(2, n_agents, 4),
        },
        batch_size=[2],
    )

    result = RuntimeInteractionContext(descriptor, mixer, inputs, outputs, batch)(batch.clone())

    assert result["chosen_action_value"].shape == (2, 1)
