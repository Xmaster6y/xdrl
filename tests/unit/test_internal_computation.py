from dataclasses import replace

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow

from xdrl.interactions import (
    InteractionContract,
    InteractionPhase,
    InternalComputationAxis,
    InternalComputationSemantics,
    InternalOccurrence,
    InternalOccurrenceSelection,
    OccurrenceIdentityError,
    RecurrentSemantics,
    RecurrentStateTransition,
    RuntimeInteractionContext,
)
from xdrl.observations import ObservationTrace, RetentionPolicy, TensorRetention
from xdrl.tdhook import TDHookWorkflowRunner
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


class RepeatedConvLSTMFixture(torch.nn.Module):
    """Tiny shared-cell fixture; it tests call identity, not ConvLSTM fidelity."""

    def __init__(self, ticks: int = 2) -> None:
        super().__init__()
        self.ticks = ticks
        self.cell = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
        torch.nn.init.constant_(self.cell.weight, 0.5)

    def forward(self, observation: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = state + observation
        for _tick in range(self.ticks):
            hidden = self.cell(hidden)  # semantic layer 0
            hidden = self.cell(hidden)  # semantic layer 1, same module instance
        return hidden, hidden.mean(dim=(-2, -1))


class AmbiguousCell(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value, value


class SingletonCell(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value,)


class SingletonOutputFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cell = SingletonCell()

    def forward(self, observation: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (hidden,) = self.cell(state + observation)
        return hidden, hidden.mean(dim=(-2, -1))


def _semantics() -> InternalComputationSemantics:
    return InternalComputationSemantics(
        axes=(
            InternalComputationAxis("tick", (0, 1)),
            InternalComputationAxis("layer", (0, 1)),
        ),
        occurrences=tuple(
            InternalOccurrence("module.cell", call_index, (tick, layer))
            for call_index, (tick, layer) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1)))
        ),
        recurrent_state_keys=(("state",), ("next", "state")),
    )


def _interaction(*, ticks: int = 2) -> RuntimeInteractionContext:
    batch_semantics = BatchSemantics(("env",))
    inputs = TensorDictSchema(
        (
            KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),
            KeySchema("state", KeyRole.STATE, KeyPresence.REQUIRED),
            KeySchema("is_init", KeyRole.TERMINATION, KeyPresence.REQUIRED),
        ),
        batch_semantics,
    )
    outputs = TensorDictSchema(
        (
            KeySchema(("next", "state"), KeyRole.STATE, KeyPresence.PRODUCED),
            KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),
        ),
        batch_semantics,
    )
    contract = InteractionContract(
        identity="repeated-convlstm:0",
        role=ModelRole.ACTOR,
        phase=InteractionPhase.EVALUATION,
        module_path="policy",
        input_schema=inputs,
        output_schema=outputs,
        recurrent=RecurrentSemantics(
            transitions=(RecurrentStateTransition(("state",), ("next", "state")),),
            reset_keys=(("is_init",),),
        ),
        internal_computation=_semantics(),
    )
    module = TensorDictModule(
        RepeatedConvLSTMFixture(ticks),
        in_keys=["observation", "state"],
        out_keys=[("next", "state"), "action"],
    )
    batch = TensorDict(
        {
            "observation": torch.ones(2, 1, 2, 2),
            "state": torch.zeros(2, 1, 2, 2),
            "is_init": torch.zeros(2, 1, dtype=torch.bool),
        },
        batch_size=[2],
    )
    trace = ObservationTrace(RetentionPolicy(tensor=TensorRetention.DETACHED))
    return RuntimeInteractionContext(contract, module, batch, observations=trace)


def test_reused_module_records_exact_layer_tick_occurrences_without_collapsing() -> None:
    interaction = _interaction()
    expected = interaction.module(interaction.representative_input.clone())["next", "state"].clone()

    with interaction, interaction.observe_internal_computation():
        result = interaction.invoke(interaction.representative_input.clone())

    internal_records = [record for record in interaction.observations.records if record.raw_call_index is not None]
    assert result["next", "state"].shape == (2, 1, 2, 2)
    assert torch.equal(result["next", "state"], expected)
    assert [record.raw_call_index for record in internal_records] == [0, 1, 2, 3]
    assert [record.internal_coordinates for record in internal_records] == [
        (("tick", 0), ("layer", 0)),
        (("tick", 0), ("layer", 1)),
        (("tick", 1), ("layer", 0)),
        (("tick", 1), ("layer", 1)),
    ]
    assert len({id(record.payload) for record in internal_records}) == 4


def test_semantic_selection_resolves_to_exact_raw_calls() -> None:
    semantics = _semantics()

    selected = semantics.select(InternalOccurrenceSelection((("tick", 1),)))

    assert [(item.module_path, item.call_index) for item in selected] == [
        ("module.cell", 2),
        ("module.cell", 3),
    ]
    with pytest.raises(OccurrenceIdentityError, match="unknown internal-computation axes"):
        semantics.select(InternalOccurrenceSelection((("environment_time", 0),)))
    with pytest.raises(OccurrenceIdentityError, match="not declared on axis"):
        semantics.select(InternalOccurrenceSelection((("tick", 3),)))

    sparse = InternalComputationSemantics(
        axes=(InternalComputationAxis("tick", (0, 1)),),
        occurrences=(InternalOccurrence("module.cell", 0, (0,)),),
        recurrent_state_keys=(("state",),),
    )
    with pytest.raises(OccurrenceIdentityError, match="matched no declared calls"):
        sparse.select(InternalOccurrenceSelection((("tick", 1),)))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: InternalComputationAxis("", (0,)), "name must be non-empty"),
        (lambda: InternalComputationAxis("tick", ()), "requires coordinates"),
        (lambda: InternalComputationAxis("tick", (object(),)), "strings or integers"),
        (lambda: InternalComputationAxis("tick", (0, 0)), "duplicate coordinates"),
        (lambda: InternalOccurrence("", 0, (0,)), "module_path must be non-empty"),
        (lambda: InternalOccurrence("cell", -1, (0,)), "non-negative integer"),
        (lambda: InternalOccurrence("cell", 0, (True,)), "coordinates must be non-empty strings or integers"),
        (lambda: InternalOccurrence("cell", 0, (1.0,)), "coordinates must be non-empty strings or integers"),
        (lambda: InternalOccurrenceSelection((("", 0),)), "axis names must be non-empty"),
        (lambda: InternalOccurrenceSelection((("tick", 0), ("tick", 1))), "duplicate axes"),
        (lambda: InternalOccurrenceSelection((("tick", True),)), "values must be non-empty strings or integers"),
        (lambda: InternalOccurrenceSelection((("tick", 1.0),)), "values must be non-empty strings or integers"),
        (lambda: InternalComputationSemantics((), (), (("state",),)), "at least one axis"),
        (
            lambda: InternalComputationSemantics(
                (InternalComputationAxis("tick", (0,)), InternalComputationAxis("tick", (0,))),
                (InternalOccurrence("cell", 0, (0, 0)),),
                (("state",),),
            ),
            "axis names must be unique",
        ),
        (
            lambda: InternalComputationSemantics((InternalComputationAxis("tick", (0,)),), (), (("state",),)),
            "require occurrence mappings",
        ),
        (
            lambda: InternalComputationSemantics(
                (InternalComputationAxis("tick", (0,)),), (InternalOccurrence("cell", 0, (0,)),), ()
            ),
            "identify related recurrent state keys",
        ),
        (
            lambda: InternalComputationSemantics(
                (InternalComputationAxis("tick", (0,)),),
                (InternalOccurrence("cell", 0, (0,)),),
                (("state",), ("state",)),
            ),
            "duplicate recurrent state keys",
        ),
        (
            lambda: InternalComputationSemantics(
                (InternalComputationAxis("tick", (0,)),),
                (InternalOccurrence("cell", 0, (0, 1)),),
                (("state",),),
            ),
            "one coordinate for every",
        ),
        (
            lambda: InternalComputationSemantics(
                (InternalComputationAxis("tick", (0,)),),
                (InternalOccurrence("cell", 0, (1,)),),
                (("state",),),
            ),
            "not declared on internal-computation axis",
        ),
        (
            lambda: InternalComputationSemantics(
                (InternalComputationAxis("tick", (0,)),),
                (InternalOccurrence("cell", 0, (0,)), InternalOccurrence("other", 0, (0,))),
                (("state",),),
            ),
            "semantic coordinates must be unique",
        ),
        (
            lambda: InternalComputationSemantics(
                (InternalComputationAxis("tick", (0, 1)),),
                (InternalOccurrence("cell", 0, (0,)), InternalOccurrence("cell", 0, (1,))),
                (("state",),),
            ),
            "raw hook call cannot identify multiple",
        ),
        (
            lambda: InternalComputationSemantics(
                (InternalComputationAxis("tick", (0, 1)),),
                (InternalOccurrence("cell", 0, (0,)), InternalOccurrence("cell", 2, (1,))),
                (("state",),),
            ),
            "call indices.*must be contiguous from zero",
        ),
    ],
)
def test_internal_computation_types_reject_invalid_identity_contracts(factory: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_ambiguous_or_changed_call_mapping_fails_and_hooks_are_cleaned_up() -> None:
    interaction = _interaction(ticks=3)
    cell = interaction.module.get_submodule("module.cell")
    original_hooks = tuple(cell._forward_hooks)

    with pytest.raises(OccurrenceIdentityError, match="undeclared internal occurrence"):
        with interaction, interaction.observe_internal_computation():
            interaction.invoke(interaction.representative_input.clone())

    assert tuple(cell._forward_hooks) == original_hooks


def test_missing_internal_calls_fail_before_root_output_is_accepted() -> None:
    interaction = _interaction(ticks=1)

    with pytest.raises(OccurrenceIdentityError, match="occurrence counts mismatch"):
        with interaction, interaction.observe_internal_computation():
            interaction.invoke(interaction.representative_input.clone())


def test_internal_observer_rejects_invalid_lifecycle_and_target_paths() -> None:
    interaction = _interaction()

    with pytest.raises(RuntimeError, match="requires an active interaction context"):
        with interaction.observe_internal_computation():
            pass

    without_semantics = RuntimeInteractionContext(
        replace(interaction.contract, internal_computation=None),
        interaction.module,
        interaction.representative_input,
    )
    with without_semantics, pytest.raises(RuntimeError, match="does not declare"):
        with without_semantics.observe_internal_computation():
            pass

    with interaction, interaction.observe_internal_computation():
        with pytest.raises(RuntimeError, match="already active"):
            with interaction.observe_internal_computation():
                pass
        with pytest.raises(OccurrenceIdentityError, match="outside the declared root"):
            interaction.module.get_submodule("module.cell")(torch.ones(1, 1, 2, 2))

    missing_path = InternalComputationSemantics(
        axes=(InternalComputationAxis("tick", (0,)),),
        occurrences=(InternalOccurrence("module.missing", 0, (0,)),),
        recurrent_state_keys=(("state",),),
    )
    missing_path_interaction = RuntimeInteractionContext(
        replace(interaction.contract, internal_computation=missing_path),
        interaction.module,
        interaction.representative_input,
    )
    with missing_path_interaction, pytest.raises(OccurrenceIdentityError, match="does not exist"):
        with missing_path_interaction.observe_internal_computation():
            pass


def test_nested_root_calls_fail_as_ambiguous() -> None:
    interaction = _interaction()
    recurse = True

    def invoke_nested_root(_module: torch.nn.Module, args: tuple[object, ...]) -> None:
        nonlocal recurse
        if recurse:
            recurse = False
            interaction.module(args[0].clone())

    handle = interaction.module.register_forward_pre_hook(invoke_nested_root)
    try:
        with pytest.raises(OccurrenceIdentityError, match="nested root calls"):
            with interaction, interaction.observe_internal_computation():
                interaction.invoke(interaction.representative_input.clone())
    finally:
        handle.remove()


def test_internal_observer_rejects_ambiguous_module_outputs() -> None:
    interaction = _interaction()
    interaction.module.module.cell = AmbiguousCell()

    with pytest.raises(OccurrenceIdentityError, match="must return one tensor"):
        with interaction, interaction.observe_internal_computation():
            interaction.invoke(interaction.representative_input.clone())


def test_internal_observer_accepts_one_tensor_inside_a_singleton_output() -> None:
    interaction = _interaction()
    semantics = InternalComputationSemantics(
        axes=(InternalComputationAxis("tick", (0,)),),
        occurrences=(InternalOccurrence("module.cell", 0, (0,)),),
        recurrent_state_keys=(("state",),),
    )
    module = TensorDictModule(
        SingletonOutputFixture(),
        in_keys=["observation", "state"],
        out_keys=[("next", "state"), "action"],
    )
    interaction = RuntimeInteractionContext(
        replace(interaction.contract, internal_computation=semantics),
        module,
        interaction.representative_input,
        observations=ObservationTrace(),
    )

    with interaction, interaction.observe_internal_computation():
        result = interaction.invoke(interaction.representative_input.clone())

    assert result["next", "state"].shape == (2, 1, 2, 2)
    assert [
        record.raw_call_index for record in interaction.observations.records if record.raw_call_index is not None
    ] == [0]


def test_tdhook_workflows_fail_before_planning_without_occurrence_evidence() -> None:
    interaction = _interaction()
    workflow = Workflow(ActivationCaching("module.cell", cache_key=("activations", "cell")))

    with pytest.raises(RuntimeError, match="occurrence-selector.*cannot guarantee identity"):
        TDHookWorkflowRunner(interaction).plan(workflow, interaction.representative_input.clone())


def test_internal_observer_rejects_root_module_and_callable_overrides() -> None:
    interaction = _interaction()
    override = TensorDictModule(
        RepeatedConvLSTMFixture(),
        in_keys=["observation", "state"],
        out_keys=[("next", "state"), "action"],
    )

    with interaction, interaction.observe_internal_computation():
        with pytest.raises(OccurrenceIdentityError, match="module overrides are unsupported"):
            interaction.invoke(interaction.representative_input.clone(), module=override)
        with pytest.raises(OccurrenceIdentityError, match="callable overrides are unsupported"):
            interaction.invoke_callable(interaction.representative_input.clone(), override)

    assert not interaction.events


def test_internal_axes_cannot_relabel_environment_time_or_unrelated_state() -> None:
    interaction = _interaction()
    contract = interaction.contract

    with pytest.raises(ValueError, match="requires recurrent semantics"):
        replace(contract, recurrent=None)

    with pytest.raises(ValueError, match="distinct from environment/sequence time"):
        recurrent_with_time = RecurrentSemantics(
            transitions=contract.recurrent.transitions,
            reset_keys=contract.recurrent.reset_keys,
            sequence_dimension="env",
        )
        internal_with_environment_axis = InternalComputationSemantics(
            axes=(InternalComputationAxis("env", (0, 1)),),
            occurrences=(
                InternalOccurrence("module.cell", 0, (0,)),
                InternalOccurrence("module.cell", 1, (1,)),
            ),
            recurrent_state_keys=(("state",),),
        )
        InteractionContract(
            identity=contract.identity,
            role=contract.role,
            phase=contract.phase,
            module_path=contract.module_path,
            input_schema=contract.input_schema,
            output_schema=contract.output_schema,
            time_dimension="env",
            recurrent=recurrent_with_time,
            internal_computation=internal_with_environment_axis,
        )

    invalid = InternalComputationSemantics(
        axes=(InternalComputationAxis("tick", (0,)),),
        occurrences=(InternalOccurrence("module.cell", 0, (0,)),),
        recurrent_state_keys=(("other",),),
    )
    with pytest.raises(ValueError, match="outside the recurrent state transitions"):
        InteractionContract(
            identity=contract.identity,
            role=contract.role,
            phase=contract.phase,
            module_path=contract.module_path,
            input_schema=contract.input_schema,
            output_schema=contract.output_schema,
            recurrent=contract.recurrent,
            internal_computation=invalid,
        )
