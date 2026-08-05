import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.targets import Target
from tdhook.workflow import Workflow

from xdrl.compatibility import WORKFLOW_CONFORMANCE, ConformanceCheck
from xdrl.interactions import InteractionContract, InteractionPhase, RuntimeInteractionContext
from xdrl.provenance import WorkflowProvenance
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
    contract = InteractionContract(
        "policy:evaluation:parity",
        ModelRole.ACTOR,
        InteractionPhase.EVALUATION,
        "policy",
        inputs,
        outputs,
    )
    return RuntimeInteractionContext(contract, policy, batch)


@pytest.mark.behavioural_parity
def test_observation_only_workflow_has_native_output_and_schema_parity() -> None:
    interaction = _interaction()
    with torch.no_grad():
        interaction.module.module.weight.copy_(torch.tensor([[2.0, -1.0]]))
    data = interaction.representative_input
    expected = interaction.module(data.clone())
    workflow = Workflow(ActivationCaching("module", cache_key=("activations", "head")))

    actual = TDHookWorkflowRunner(interaction).run(workflow, data.clone(), code_revision="test-revision")

    assert actual.data.batch_size == expected.batch_size
    assert torch.equal(actual.data.get("action"), expected.get("action"))
    assert [event.kind.value for event in actual.provenance.lifecycle_events] == ["before", "after"]
    assert WorkflowProvenance.from_json(actual.provenance.to_json()) == actual.provenance
    assert not interaction.module.module._forward_hooks


@pytest.mark.behavioural_parity
def test_workflow_cleanup_is_exception_safe() -> None:
    interaction = _interaction()
    runner = TDHookWorkflowRunner(interaction)
    workflow = Workflow(ActivationCaching(Target("missing", "activation", -1, (0,))))

    with pytest.raises(ValueError):
        runner.run(workflow, interaction.representative_input.clone(), code_revision="test-revision")

    assert not interaction.module.module._forward_hooks


@pytest.mark.behavioural_parity
def test_lazy_materialisation_is_explicit_before_workflow_execution() -> None:
    interaction = _interaction(lazy=True)
    runner = TDHookWorkflowRunner(interaction)
    workflow = Workflow(ActivationCaching("module"))

    with pytest.raises(RuntimeError, match="materialize"):
        runner.run(workflow, interaction.representative_input.clone(), code_revision="test-revision")
    runner.materialize()
    expected = interaction.module(interaction.representative_input.clone())
    actual = runner.run(workflow, interaction.representative_input.clone(), code_revision="test-revision")

    assert torch.equal(actual.data.get("action"), expected.get("action"))
    assert set(WORKFLOW_CONFORMANCE[0].checks) == set(ConformanceCheck)
