import json

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow
from torchrl.data import UnboundedContinuous

from xdrl.interactions import (
    AgentSelector,
    CoalitionTerm,
    InteractionContract,
    InteractionPhase,
    InteractionTopology,
    MultiAgentSemantics,
    NamedReduction,
    RuntimeInteractionContext,
    SemanticTarget,
    ValueDecompositionAxes,
    ValueDecompositionKeys,
    ValueDecompositionSemantics,
)
from xdrl.provenance import WorkflowProvenance
from xdrl.tdhook import TDHookWorkflowRunner
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


AGENTS = ("alice", "bob", "carol")
KEYS = ValueDecompositionKeys(
    individual_value=("agents", "individual_value"),
    coalition_contribution=("value_decomposition", "coalition_contribution"),
    semantic_mask=("value_decomposition", "semantic_mask"),
    mixer_input=("mixer", "state"),
    joint_value=("mixer", "joint_value"),
)
AXES = ValueDecompositionAxes(
    individual_value=("env", "agent", "value_feature"),
    coalition_contribution=("env", "coalition", "value_feature"),
    semantic_mask=("env", "coalition", "agent"),
    mixer_input=("env", "state_feature"),
    joint_value=("env", "value_feature"),
)
TERMS = (
    CoalitionTerm("alice", ("alice",), 0),
    CoalitionTerm("bob", ("bob",), 1),
    CoalitionTerm("carol", ("carol",), 2),
    CoalitionTerm("alice+bob", ("alice", "bob"), 3),
    CoalitionTerm("alice+carol", ("alice", "carol"), 4),
    CoalitionTerm("bob+carol", ("bob", "carol"), 5),
    CoalitionTerm("alice+bob+carol", AGENTS, 6),
)
REDUCTION = NamedReduction(
    "sum_coalition_contributions",
    KEYS.coalition_contribution,
    KEYS.joint_value,
    ("coalition",),
)


class _DeterministicCoalitionMixer(torch.nn.Module):
    """A fixture whose one module serves every semantic agent and coalition."""

    def forward(
        self,
        individual_value: torch.Tensor,
        semantic_mask: torch.Tensor,
        mixer_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contributions = torch.einsum("eca,eav->ecv", semantic_mask, individual_value)
        contributions = contributions + mixer_input.sum(dim=-1, keepdim=True).unsqueeze(-2)
        return contributions, contributions.sum(dim=-2)


def _schemas() -> tuple[TensorDictSchema, TensorDictSchema]:
    batch = BatchSemantics(("env",))
    return (
        TensorDictSchema(
            (
                KeySchema(
                    KEYS.individual_value,
                    KeyRole.INDIVIDUAL_VALUE,
                    KeyPresence.REQUIRED,
                    UnboundedContinuous(shape=(len(AGENTS), 1)),
                ),
                KeySchema(
                    KEYS.semantic_mask,
                    KeyRole.SEMANTIC_MASK,
                    KeyPresence.REQUIRED,
                    UnboundedContinuous(shape=(len(TERMS), len(AGENTS))),
                ),
                KeySchema(
                    KEYS.mixer_input,
                    KeyRole.MIXER_INPUT,
                    KeyPresence.REQUIRED,
                    UnboundedContinuous(shape=(2,)),
                ),
            ),
            batch,
        ),
        TensorDictSchema(
            (
                KeySchema(
                    KEYS.coalition_contribution,
                    KeyRole.COALITION_CONTRIBUTION,
                    KeyPresence.PRODUCED,
                    UnboundedContinuous(shape=(len(TERMS), 1)),
                ),
                KeySchema(
                    KEYS.joint_value,
                    KeyRole.JOINT_VALUE,
                    KeyPresence.PRODUCED,
                    UnboundedContinuous(shape=(1,)),
                ),
            ),
            batch,
        ),
    )


def _contract(
    *,
    terms: tuple[CoalitionTerm, ...] = TERMS,
    reductions: tuple[NamedReduction, ...] = (REDUCTION,),
    axes: ValueDecompositionAxes = AXES,
) -> InteractionContract:
    inputs, outputs = _schemas()
    multi_agent = MultiAgentSemantics(
        InteractionTopology.MIXER,
        "agents",
        len(AGENTS),
        SemanticTarget(ModelRole.MIXER, AgentSelector("agents", ("alice", "carol"))),
        agent_identities=AGENTS,
    )
    decomposition = ValueDecompositionSemantics(
        coalition_axis="coalition",
        feature_axes=("value_feature", "state_feature"),
        terms=terms,
        keys=KEYS,
        axes=axes,
        reductions=reductions,
        parameters_shared=True,
        coalition_targets=tuple(
            identity
            for identity in ("alice+carol", "alice+bob+carol")
            if identity in {term.identity for term in terms}
        ),
    )
    return InteractionContract(
        identity="shared-mixer:0",
        role=ModelRole.MIXER,
        phase=InteractionPhase.EVALUATION,
        module_path="loss.shared_mixer",
        input_schema=inputs,
        output_schema=outputs,
        agent_dimension="agent",
        multi_agent=multi_agent,
        value_decomposition=decomposition,
    )


def test_deterministic_parameter_shared_mixer_preserves_semantic_coalitions() -> None:
    contract = _contract()
    module = TensorDictModule(
        _DeterministicCoalitionMixer(),
        in_keys=[KEYS.individual_value, KEYS.semantic_mask, KEYS.mixer_input],
        out_keys=[KEYS.coalition_contribution, KEYS.joint_value],
    )
    semantic_mask = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=torch.float,
    ).expand(2, -1, -1)
    batch = TensorDict(
        {
            KEYS.individual_value: torch.tensor([[[1.0], [2.0], [4.0]], [[2.0], [3.0], [5.0]]]),
            KEYS.semantic_mask: semantic_mask,
            KEYS.mixer_input: torch.zeros(2, 2),
        },
        batch_size=[2],
    )

    result = RuntimeInteractionContext(contract, module, batch)(batch.clone())

    assert result[KEYS.coalition_contribution][0, :, 0].tolist() == [1, 2, 4, 3, 5, 6, 7]
    assert result[KEYS.joint_value][:, 0].tolist() == [28, 40]
    assert contract.module_path == "loss.shared_mixer"
    assert contract.value_decomposition.parameters_shared
    assert tuple(term.identity for term in contract.value_decomposition.targeted_terms) == (
        "alice+carol",
        "alice+bob+carol",
    )
    assert tuple(term.identity for term in contract.value_decomposition.terms) != (contract.module_path,)


def test_value_decomposition_is_json_serialisable_without_flattening_keys_or_axes() -> None:
    encoded = json.loads(json.dumps(_contract().to_dict()))["value_decomposition"]

    assert encoded["terms"][3] == {"identity": "alice+bob", "members": ["alice", "bob"], "axis_index": 3}
    assert encoded["keys"]["semantic_mask"] == ["value_decomposition", "semantic_mask"]
    assert encoded["axes"]["coalition_contribution"] == ["env", "coalition", "value_feature"]
    assert encoded["reductions"][0]["name"] == "sum_coalition_contributions"
    assert encoded["coalition_targets"] == ["alice+carol", "alice+bob+carol"]


def test_named_reduction_and_coalitions_round_trip_in_workflow_provenance() -> None:
    contract = _contract()
    module = TensorDictModule(
        _DeterministicCoalitionMixer(),
        in_keys=[KEYS.individual_value, KEYS.semantic_mask, KEYS.mixer_input],
        out_keys=[KEYS.coalition_contribution, KEYS.joint_value],
    )
    batch = TensorDict(
        {
            KEYS.individual_value: torch.ones(1, len(AGENTS), 1),
            KEYS.semantic_mask: torch.ones(1, len(TERMS), len(AGENTS)),
            KEYS.mixer_input: torch.zeros(1, 2),
        },
        batch_size=[1],
    )
    context = RuntimeInteractionContext(contract, module, batch)

    def coalition_output(output: tuple[torch.Tensor, torch.Tensor], **_kwargs: object) -> torch.Tensor:
        return output[0]

    execution = TDHookWorkflowRunner(context).run(
        Workflow(ActivationCaching("module", callback=coalition_output)),
        batch.clone(),
        code_revision="test-revision",
        callback_identifiers={coalition_output: "coalition-output-v1"},
    )
    restored = WorkflowProvenance.from_json(execution.provenance.to_json())

    decomposition = restored.interaction_contract["value_decomposition"]
    assert decomposition["terms"][-1]["members"] == AGENTS
    assert decomposition["reductions"][0]["reduced_axes"] == ("coalition",)


def test_coalition_membership_is_unique_ordered_and_within_the_declared_group() -> None:
    with pytest.raises(ValueError, match="duplicate agents"):
        CoalitionTerm("bad", ("alice", "alice"), 0)
    with pytest.raises(ValueError, match="members must follow declared agent order"):
        _contract(terms=(CoalitionTerm("bad-order", ("bob", "alice"), 0),))
    with pytest.raises(ValueError, match="outside the declared group"):
        _contract(terms=(CoalitionTerm("outsider", ("dave",), 0),))
    with pytest.raises(ValueError, match="memberships must be unique"):
        _contract(
            terms=(
                CoalitionTerm("first", ("alice",), 0),
                CoalitionTerm("duplicate", ("alice",), 1),
            )
        )


def test_implicit_or_misdirected_joint_value_aggregation_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one named reduction"):
        _contract(reductions=())
    with pytest.raises(ValueError, match="joint_value requires a named reduction"):
        _contract(
            reductions=(
                NamedReduction(
                    "wrong-axis",
                    KEYS.semantic_mask,
                    KEYS.joint_value,
                    ("agent",),
                ),
            )
        )


def test_coalition_axis_is_distinct_and_required_on_coalition_tensors() -> None:
    colliding = ValueDecompositionSemantics(
        coalition_axis="agent",
        feature_axes=("value_feature", "state_feature"),
        terms=TERMS,
        keys=KEYS,
        axes=AXES,
        reductions=(REDUCTION,),
        parameters_shared=True,
        coalition_targets=("alice+carol",),
    )
    inputs, outputs = _schemas()
    multi_agent = MultiAgentSemantics(
        InteractionTopology.MIXER,
        "agents",
        len(AGENTS),
        SemanticTarget(ModelRole.MIXER, AgentSelector("agents")),
        agent_identities=AGENTS,
    )

    with pytest.raises(ValueError, match="distinct from environment, time, agent, and feature"):
        InteractionContract(
            "bad-axes",
            ModelRole.MIXER,
            InteractionPhase.EVALUATION,
            "loss.shared_mixer",
            inputs,
            outputs,
            agent_dimension="agent",
            multi_agent=multi_agent,
            value_decomposition=colliding,
        )
