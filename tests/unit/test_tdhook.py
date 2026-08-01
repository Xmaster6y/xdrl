import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent.activation_caching import ActivationCaching

from xdrl.interactions import InteractionDescriptor, InteractionPhase, RuntimeInteractionContext, SchemaSnapshot
from xdrl.tdhook import TDHookInteractionAdapter
from xdrl.types import BatchSemantics, KeyPresence, KeyRole, KeySchema, ModelRole, TensorDictSchema


def _schemas() -> tuple[TensorDictSchema, TensorDictSchema]:
    return (
        TensorDictSchema(
            (KeySchema(("agents", "observation"), KeyRole.OBSERVATION, KeyPresence.REQUIRED),),
            BatchSemantics(("env", "agent")),
        ),
        TensorDictSchema(
            (KeySchema(("agents", "action"), KeyRole.ACTION, KeyPresence.PRODUCED),),
            BatchSemantics(("env", "agent")),
        ),
    )


def _policy(lazy: bool = False) -> TensorDictModule:
    layer: torch.nn.Module = torch.nn.LazyLinear(1, bias=False) if lazy else torch.nn.Linear(2, 1, bias=False)
    return TensorDictModule(layer, in_keys=[("agents", "observation")], out_keys=[("agents", "action")])


def _interaction(policy: TensorDictModule) -> RuntimeInteractionContext:
    inputs, outputs = _schemas()
    batch = TensorDict({("agents", "observation"): torch.ones(3, 2, 2)}, batch_size=[3, 2])
    descriptor = InteractionDescriptor(
        "policy:evaluation:0",
        ModelRole.ACTOR,
        InteractionPhase.EVALUATION,
        "policy",
        SchemaSnapshot.from_schema(inputs),
        SchemaSnapshot.from_schema(outputs),
        batch_dimensions=("env", "agent"),
    )
    return RuntimeInteractionContext(descriptor, policy, inputs, outputs, batch)


def test_activation_cache_preserves_nested_batched_tensordict_execution() -> None:
    policy = _policy()
    with torch.no_grad():
        policy.module.weight.copy_(torch.tensor([[2.0, -1.0]]))
    interaction = _interaction(policy)
    adapter = TDHookInteractionAdapter(interaction, aliases={"policy_head": "module"})
    batch = interaction.representative_input.clone()
    expected = policy(batch.clone()).get(("agents", "action")).clone()
    factory = ActivationCaching(r"module")

    with adapter.activate(factory) as active:
        result = active.invoke(batch)
        assert torch.equal(result.get(("agents", "action")), expected)
        assert active.target_paths == {"policy_head": "td_module.module"}
        assert "module" in active.module_paths
        assert "module" in active.contexts[0].cache.keys()

    assert not policy.module._forward_hooks


def test_adapter_rejects_unresolved_paths_and_contract_mismatches() -> None:
    interaction = _interaction(_policy())
    with pytest.raises(ValueError, match="cannot resolve"):
        TDHookInteractionAdapter(interaction, aliases={"missing": "module.missing"})
    with pytest.raises(ValueError, match="selected input keys"):
        TDHookInteractionAdapter(interaction, input_keys=["other"])
    with pytest.raises(ValueError, match="interaction output contract"):
        TDHookInteractionAdapter(interaction, output_keys=[])


def test_lazy_modules_require_explicit_materialisation() -> None:
    policy = _policy(lazy=True)
    interaction = _interaction(policy)
    adapter = TDHookInteractionAdapter(interaction)
    factory = ActivationCaching(r"module")

    with pytest.raises(RuntimeError, match="call materialize"):
        adapter.activate(factory)
    adapter.materialize()
    with adapter.activate(factory):
        adapter.invoke(interaction.representative_input.clone())


def test_adapter_removes_hooks_when_invocation_fails() -> None:
    policy = _policy()
    adapter = TDHookInteractionAdapter(_interaction(policy))
    factory = ActivationCaching(r"module")
    with pytest.raises(RuntimeError, match="boom"):
        with adapter.activate(factory):
            raise RuntimeError("boom")
    assert not policy.module._forward_hooks
