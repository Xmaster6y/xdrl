import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent.activation_caching import ActivationCaching

from xdrl.compatibility import ADAPTER_CONFORMANCE, ConformanceCheck
from xdrl.interactions import InteractionDescriptor, InteractionPhase, RuntimeInteractionContext, SchemaSnapshot
from xdrl.tdhook import TDHookInteractionAdapter
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
        module_aliases={"head": "module"},
    )
    return RuntimeInteractionContext(descriptor, policy, inputs, outputs, batch)


@pytest.mark.behavioural_parity
def test_observation_only_adapter_has_deterministic_output_and_schema_parity() -> None:
    interaction = _interaction()
    with torch.no_grad():
        interaction.module.module.weight.copy_(torch.tensor([[2.0, -1.0]]))
    batch = interaction.representative_input
    expected = interaction.module(batch.clone())
    adapter = TDHookInteractionAdapter(interaction)

    with adapter.activate(ActivationCaching(r"module")) as active:
        actual = active.invoke(batch.clone())
        assert "module" in active.contexts[0].cache

    assert actual.batch_size == expected.batch_size
    assert set(actual.keys(include_nested=True, leaves_only=True)) == set(
        expected.keys(include_nested=True, leaves_only=True)
    )
    assert torch.equal(actual.get("action"), expected.get("action"))
    assert [event.kind.value for event in interaction.events] == ["before", "after"]
    assert not interaction.module.module._forward_hooks


@pytest.mark.behavioural_parity
def test_adapter_cleanup_is_exception_safe_and_paths_are_resolved() -> None:
    interaction = _interaction()
    adapter = TDHookInteractionAdapter(interaction)

    assert adapter.target_paths == {"head": "td_module.module"}
    with pytest.raises(RuntimeError, match="stop"):
        with adapter.activate(ActivationCaching(r"module")):
            raise RuntimeError("stop")

    assert not interaction.module.module._forward_hooks
    assert adapter.contexts == ()


@pytest.mark.behavioural_parity
def test_lazy_materialisation_is_explicit_before_parity_execution() -> None:
    interaction = _interaction(lazy=True)
    adapter = TDHookInteractionAdapter(interaction)

    with pytest.raises(RuntimeError, match="materialize"):
        adapter.activate(ActivationCaching(r"module"))
    adapter.materialize()
    expected = interaction.module(interaction.representative_input.clone())
    with adapter.activate(ActivationCaching(r"module")) as active:
        actual = active.invoke(interaction.representative_input.clone())

    assert torch.equal(actual.get("action"), expected.get("action"))
    assert set(ADAPTER_CONFORMANCE[0].checks) == set(ConformanceCheck)
