import json
from collections.abc import Callable
from dataclasses import replace

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
from xdrl.provenance import ProvenanceSchemaError, WorkflowProvenance
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


def _value_decomposition_provenance() -> WorkflowProvenance:
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

    return (
        TDHookWorkflowRunner(context)
        .run(
            Workflow(ActivationCaching("module", callback=coalition_output)),
            batch.clone(),
            code_revision="test-revision",
            callback_identifiers={coalition_output: "coalition-output-v1"},
        )
        .provenance
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
    assert contract.module_path not in {term.identity for term in contract.value_decomposition.terms}


def test_value_decomposition_is_json_serialisable_without_flattening_keys_or_axes() -> None:
    encoded = json.loads(json.dumps(_contract().to_dict()))["value_decomposition"]

    assert encoded["terms"][3] == {"identity": "alice+bob", "members": ["alice", "bob"], "axis_index": 3}
    assert encoded["keys"]["semantic_mask"] == ["value_decomposition", "semantic_mask"]
    assert encoded["axes"]["coalition_contribution"] == ["env", "coalition", "value_feature"]
    assert encoded["reductions"][0]["name"] == "sum_coalition_contributions"
    assert encoded["coalition_targets"] == ["alice+carol", "alice+bob+carol"]


def test_named_reduction_and_coalitions_round_trip_in_workflow_provenance() -> None:
    restored = WorkflowProvenance.from_json(_value_decomposition_provenance().to_json())

    decomposition = restored.interaction_contract["value_decomposition"]
    assert decomposition["terms"][-1]["members"] == AGENTS
    assert decomposition["reductions"][0]["reduced_axes"] == ("coalition",)


def test_coalition_membership_is_unique_ordered_and_within_the_declared_group() -> None:
    with pytest.raises(TypeError, match="identity must be a string"):
        CoalitionTerm(5, ("alice",), 0)  # type: ignore[arg-type]
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


def test_implicit_agent_identifiers_reject_strings_and_booleans() -> None:
    for invalid in ("alice", True):
        with pytest.raises(ValueError, match="outside the declared group"):
            MultiAgentSemantics(
                InteractionTopology.MIXER,
                "agents",
                3,
                SemanticTarget(ModelRole.MIXER, AgentSelector("agents", (invalid,))),
            )


def test_value_axes_require_the_ordered_batch_prefix() -> None:
    reordered = replace(AXES, individual_value=("agent", "env", "value_feature"))

    with pytest.raises(ValueError, match="must begin with the declared batch dimensions in order"):
        _contract(axes=reordered)


@pytest.mark.parametrize(
    ("key_name", "shape", "message"),
    [
        ("individual_value", (4, 1), "individual_value agent extent must be 3"),
        ("coalition_contribution", (6, 1), "coalition_contribution coalition extent must be 7"),
        ("semantic_mask", (7, 4), "semantic_mask agent extent must be 3"),
    ],
)
def test_semantic_axis_extents_match_declared_agents_and_terms(
    key_name: str, shape: tuple[int, ...], message: str
) -> None:
    contract = _contract()
    schema_name = "output_schema" if key_name == "coalition_contribution" else "input_schema"
    schema = getattr(contract, schema_name)
    entries = tuple(
        replace(entry, spec=UnboundedContinuous(shape=shape)) if entry.key == getattr(KEYS, key_name) else entry
        for entry in schema.keys
    )

    with pytest.raises(ValueError, match=message):
        replace(contract, **{schema_name: replace(schema, keys=entries)})


def test_joint_value_reduction_must_start_from_coalition_contributions() -> None:
    alternate_axes = replace(
        AXES,
        semantic_mask=("env", "coalition", "value_feature"),
    )
    alternate = NamedReduction(
        "mask-to-joint",
        KEYS.semantic_mask,
        KEYS.joint_value,
        ("coalition",),
    )

    with pytest.raises(ValueError, match="from coalition_contribution"):
        _contract(axes=alternate_axes, reductions=(alternate,))


def test_flat_value_decomposition_keys_round_trip_through_provenance() -> None:
    keys = ValueDecompositionKeys(
        individual_value="individual_value",
        coalition_contribution="coalition_contribution",
        semantic_mask="semantic_mask",
        mixer_input="mixer_input",
        joint_value="joint_value",
    )
    batch_semantics = BatchSemantics(("env",))
    inputs = TensorDictSchema(
        (
            KeySchema(
                keys.individual_value,
                KeyRole.INDIVIDUAL_VALUE,
                KeyPresence.REQUIRED,
                UnboundedContinuous(shape=(len(AGENTS), 1)),
            ),
            KeySchema(
                keys.semantic_mask,
                KeyRole.SEMANTIC_MASK,
                KeyPresence.REQUIRED,
                UnboundedContinuous(shape=(len(TERMS), len(AGENTS))),
            ),
            KeySchema(
                keys.mixer_input,
                KeyRole.MIXER_INPUT,
                KeyPresence.REQUIRED,
                UnboundedContinuous(shape=(2,)),
            ),
        ),
        batch_semantics,
    )
    outputs = TensorDictSchema(
        (
            KeySchema(
                keys.coalition_contribution,
                KeyRole.COALITION_CONTRIBUTION,
                KeyPresence.PRODUCED,
                UnboundedContinuous(shape=(len(TERMS), 1)),
            ),
            KeySchema(
                keys.joint_value,
                KeyRole.JOINT_VALUE,
                KeyPresence.PRODUCED,
                UnboundedContinuous(shape=(1,)),
            ),
        ),
        batch_semantics,
    )
    reduction = NamedReduction(
        "sum-coalitions",
        keys.coalition_contribution,
        keys.joint_value,
        ("coalition",),
    )
    decomposition = ValueDecompositionSemantics(
        "coalition",
        ("value_feature", "state_feature"),
        TERMS,
        keys,
        AXES,
        (reduction,),
    )
    multi_agent = MultiAgentSemantics(
        InteractionTopology.MIXER,
        "agents",
        len(AGENTS),
        SemanticTarget(ModelRole.MIXER, AgentSelector("agents")),
        agent_identities=AGENTS,
    )
    contract = InteractionContract(
        "flat-keys",
        ModelRole.MIXER,
        InteractionPhase.EVALUATION,
        "loss.shared_mixer",
        inputs,
        outputs,
        agent_dimension="agent",
        multi_agent=multi_agent,
        value_decomposition=decomposition,
    )
    module = TensorDictModule(
        _DeterministicCoalitionMixer(),
        in_keys=[keys.individual_value, keys.semantic_mask, keys.mixer_input],
        out_keys=[keys.coalition_contribution, keys.joint_value],
    )
    data = TensorDict(
        {
            keys.individual_value: torch.ones(1, len(AGENTS), 1),
            keys.semantic_mask: torch.ones(1, len(TERMS), len(AGENTS)),
            keys.mixer_input: torch.zeros(1, 2),
        },
        batch_size=[1],
    )
    context = RuntimeInteractionContext(contract, module, data)

    def coalition_output(output: tuple[torch.Tensor, torch.Tensor], **_kwargs: object) -> torch.Tensor:
        return output[0]

    execution = TDHookWorkflowRunner(context).run(
        Workflow(ActivationCaching("module", callback=coalition_output)),
        data,
        code_revision="test-revision",
        callback_identifiers={coalition_output: "flat-coalition-output-v1"},
    )
    restored = WorkflowProvenance.from_json(execution.provenance.to_json())

    assert restored.interaction_contract["value_decomposition"]["keys"]["joint_value"] == ("joint_value",)


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


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: CoalitionTerm("", ("alice",), 0), "identity must be non-empty"),
        (lambda: CoalitionTerm("empty", (), 0), "membership must be non-empty"),
        (lambda: CoalitionTerm("bad-member", ("",), 0), "members must be non-empty"),
        (lambda: CoalitionTerm("bad-index", ("alice",), -1), "axis_index must be"),
        (lambda: replace(KEYS, joint_value=""), "keys must be non-empty"),
        (lambda: replace(KEYS, joint_value=KEYS.mixer_input), "keys must be unique"),
        (lambda: replace(AXES, joint_value=()), "axes must be explicit"),
        (lambda: replace(AXES, joint_value=("env", "")), "axes must be non-empty"),
        (lambda: replace(AXES, joint_value=("env", "env")), "axes contain duplicates"),
        (lambda: NamedReduction("", "source", "target", ("agent",)), "name must be non-empty"),
        (lambda: NamedReduction("empty", "source", "target", ()), "reduced_axes must be explicit"),
        (
            lambda: NamedReduction("duplicate", "source", "target", ("agent", "agent")),
            "reduced_axes contains duplicates",
        ),
        (lambda: NamedReduction("same", "value", "value", ("agent",)), "source and target keys must differ"),
    ],
)
def test_value_decomposition_value_objects_reject_malformed_declarations(
    factory: Callable[[], object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"coalition_axis": ""}, "coalition_axis must be non-empty"),
        ({"feature_axes": ("",)}, "feature axes must be non-empty and unique"),
        ({"terms": ()}, "requires coalition terms"),
        (
            {
                "terms": (
                    CoalitionTerm("same", ("alice",), 0),
                    CoalitionTerm("same", ("bob",), 1),
                )
            },
            "coalition identities must be unique",
        ),
        ({"terms": (CoalitionTerm("offset", ("alice",), 1),)}, "indices must be contiguous"),
        (
            {"reductions": (REDUCTION, replace(REDUCTION))},
            "reduction names must be unique",
        ),
        ({"parameters_shared": 1}, "parameters_shared must be a boolean"),
        ({"coalition_targets": ("alice", "alice")}, "coalition targets must be unique"),
        ({"coalition_targets": ("unknown",)}, "unknown coalition targets"),
    ],
)
def test_value_decomposition_semantics_reject_invalid_metadata(changes: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        replace(_contract().value_decomposition, **changes)


def test_empty_coalition_target_selects_every_declared_term() -> None:
    semantics = replace(_contract().value_decomposition, coalition_targets=())

    assert semantics.targeted_terms == TERMS


@pytest.mark.parametrize(
    ("identities", "selector", "message"),
    [
        (("alice", "bob"), (), "must match the declared group size"),
        (("alice", "alice", "carol"), (), "must be unique"),
        (("alice", "", "carol"), (), "must be non-empty strings or integers"),
        (AGENTS, ("outsider",), "outside the declared agent group"),
    ],
)
def test_explicit_agent_identities_are_strictly_validated(
    identities: tuple[str, ...], selector: tuple[str, ...], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        MultiAgentSemantics(
            InteractionTopology.MIXER,
            "agents",
            3,
            SemanticTarget(ModelRole.MIXER, AgentSelector("agents", selector)),
            agent_identities=identities,
        )


def test_value_decomposition_requires_mixer_and_separate_semantic_axes() -> None:
    contract = _contract()
    with pytest.raises(ValueError, match="requires multi-agent semantics"):
        replace(contract, multi_agent=None)
    with pytest.raises(ValueError, match="requires a mixer interaction"):
        replace(contract, multi_agent=replace(contract.multi_agent, topology=InteractionTopology.PARAMETER_SHARED))

    batched_agent = BatchSemantics(("env", "agent"))
    with pytest.raises(ValueError, match="distinct from leading batch dimensions"):
        replace(
            contract,
            input_schema=replace(contract.input_schema, batch=batched_agent),
            output_schema=replace(contract.output_schema, batch=batched_agent),
        )
    with pytest.raises(ValueError, match="feature axes must be distinct"):
        replace(
            contract,
            value_decomposition=replace(contract.value_decomposition, feature_axes=("env", "value_feature")),
        )


@pytest.mark.parametrize(
    ("axes", "message"),
    [
        (replace(AXES, mixer_input=("env", "unknown")), "declares unknown axes"),
        (replace(AXES, individual_value=("env", "value_feature")), "must retain the agent dimension"),
        (replace(AXES, semantic_mask=("env", "agent", "value_feature")), "must retain the coalition axis"),
    ],
)
def test_value_decomposition_rejects_incomplete_axis_semantics(axes: ValueDecompositionAxes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _contract(axes=axes)


def test_value_decomposition_schema_keys_require_roles_specs_and_matching_rank() -> None:
    contract = _contract()
    individual = contract.input_schema.keys[0]
    without_individual = replace(contract.input_schema, keys=contract.input_schema.keys[1:])
    with pytest.raises(ValueError, match="individual_value key is not declared"):
        replace(contract, input_schema=without_individual)

    wrong_role = replace(
        contract.input_schema, keys=(replace(individual, role=KeyRole.VALUE), *contract.input_schema.keys[1:])
    )
    with pytest.raises(ValueError, match="must use the individual_value role"):
        replace(contract, input_schema=wrong_role)

    missing_spec = replace(
        contract.input_schema, keys=(replace(individual, spec=None), *contract.input_schema.keys[1:])
    )
    with pytest.raises(ValueError, match="requires a spec to validate semantic-axis extents"):
        replace(contract, input_schema=missing_spec)

    short_axes = replace(AXES, individual_value=("env", "agent"))
    with pytest.raises(ValueError, match="axes do not match its declared tensor rank"):
        _contract(axes=short_axes)


def test_named_reductions_are_bound_to_declared_keys_and_axis_transformations() -> None:
    valid = REDUCTION
    cases = (
        (
            NamedReduction("unknown-key", "unknown", KEYS.mixer_input, ("agent",)),
            "must reference declared value-decomposition keys",
        ),
        (
            NamedReduction("unknown-axis", KEYS.individual_value, KEYS.mixer_input, ("unknown",)),
            "references unknown axes",
        ),
        (
            NamedReduction("missing-source-axis", KEYS.semantic_mask, KEYS.mixer_input, ("value_feature",)),
            "axes must exist on its source tensor",
        ),
        (
            NamedReduction("wrong-target-axes", KEYS.individual_value, KEYS.mixer_input, ("agent",)),
            "target axes must equal its unreduced source axes",
        ),
    )
    for reduction, message in cases:
        with pytest.raises(ValueError, match=message):
            _contract(reductions=(valid, reduction))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"terms": {}}), "terms and.*reductions must be arrays"),
        (lambda value: value["terms"][0].update({"members": ()}), r"terms\[0\].members must be an array"),
    ],
)
def test_workflow_provenance_strictly_decodes_value_decomposition(
    mutation: Callable[[dict[str, object]], None], message: str
) -> None:
    payload = _value_decomposition_provenance().to_dict()
    decomposition = payload["interaction_contract"]["value_decomposition"]
    mutation(decomposition)

    with pytest.raises(ProvenanceSchemaError, match=message):
        WorkflowProvenance.from_dict(payload)
