import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.targets import Target
from tdhook.workflow import Workflow

from xdrl.compatibility import WORKFLOW_CONFORMANCE, ConformanceCheck
from xdrl.interactions import InteractionDescriptor, InteractionPhase, RuntimeInteractionContext, SchemaSnapshot
from xdrl.tdhook import TDHookWorkflowRunner
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


def _interaction(*, lazy: bool = False) -> RuntimeInteractionContext:
    inputs = TensorDictSchema(
        (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),), BatchSemantics(("env",))
    )
    outputs = TensorDictSchema((KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),), BatchSemantics(("env",)))
    layer: torch.nn.Module = torch.nn.LazyLinear(1, bias=False) if lazy else torch.nn.Linear(2, 1, bias=False)
    policy = TensorDictModule(layer, in_keys=["observation"], out_keys=["action"])
    batch = TensorDict({"observation": torch.tensor([[2.0, -1.0], [0.5, 3.0]])}, batch_size=[2])
    descriptor = InteractionDescriptor(
        "policy:evaluation:parity",
        ModelRole.ACTOR,
        InteractionPhase.EVALUATION,
        "policy",
        SchemaSnapshot.from_schema(inputs),
        SchemaSnapshot.from_schema(outputs),
        batch_dimensions=("env",),
    )
    return RuntimeInteractionContext(descriptor, policy, inputs, outputs, batch)


@pytest.mark.behavioural_parity
def test_observation_only_workflow_has_native_output_and_schema_parity() -> None:
    interaction = _interaction()
    with torch.no_grad():
        interaction.module.module.weight.copy_(torch.tensor([[2.0, -1.0]]))
    data = interaction.representative_input
    expected = interaction.module(data.clone())
    workflow = Workflow(ActivationCaching("module", cache_key=("activations", "head")))

    actual = TDHookWorkflowRunner(interaction).run(workflow, data.clone())

    assert actual.data.batch_size == expected.batch_size
    assert torch.equal(actual.data.get("action"), expected.get("action"))
    assert [event.kind.value for event in actual.record.events] == ["before", "after"]
    assert not interaction.module.module._forward_hooks


@pytest.mark.behavioural_parity
def test_workflow_cleanup_is_exception_safe() -> None:
    interaction = _interaction()
    runner = TDHookWorkflowRunner(interaction)
    workflow = Workflow(ActivationCaching(Target("missing", "activation", -1, (0,))))

    with pytest.raises(ValueError):
        runner.run(workflow, interaction.representative_input.clone())

    assert not interaction.module.module._forward_hooks


@pytest.mark.behavioural_parity
def test_lazy_materialisation_is_explicit_before_workflow_execution() -> None:
    interaction = _interaction(lazy=True)
    runner = TDHookWorkflowRunner(interaction)
    workflow = Workflow(ActivationCaching("module"))

    with pytest.raises(RuntimeError, match="materialize"):
        runner.run(workflow, interaction.representative_input.clone())
    runner.materialize()
    expected = interaction.module(interaction.representative_input.clone())
    actual = runner.run(workflow, interaction.representative_input.clone())

    assert torch.equal(actual.data.get("action"), expected.get("action"))
    assert set(WORKFLOW_CONFORMANCE[0].checks) == set(ConformanceCheck)
