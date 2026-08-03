from dataclasses import replace

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent.activation_caching import ActivationCaching

from xdrl.interactions import InteractionDescriptor, InteractionPhase, RuntimeInteractionContext, SchemaSnapshot
from xdrl.interventions import (
    Intervention,
    InterventionTarget,
    InterventionTiming,
    InterventionValidationError,
    TDHookInterventionFactory,
)
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


def test_lazy_buffers_require_explicit_materialisation() -> None:
    policy = TensorDictModule(
        torch.nn.LazyBatchNorm1d(affine=False),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "action")],
    )
    interaction = _interaction(policy)
    adapter = TDHookInteractionAdapter(interaction)
    factory = ActivationCaching(r"module")

    with pytest.raises(RuntimeError, match="call materialize"):
        adapter.activate(factory)
    adapter.materialize()
    with adapter.activate(factory):
        adapter.invoke(interaction.representative_input.clone())


def test_adapter_rejects_compiled_descendants() -> None:
    policy = _policy()
    policy.module._orig_mod = policy.module
    with pytest.raises(NotImplementedError, match="torch.compile"):
        TDHookInteractionAdapter(_interaction(policy))


def test_adapter_removes_hooks_when_invocation_fails() -> None:
    policy = _policy()
    adapter = TDHookInteractionAdapter(_interaction(policy))
    factory = ActivationCaching(r"module")
    with pytest.raises(RuntimeError, match="boom"):
        with adapter.activate(factory):
            raise RuntimeError("boom")
    assert not policy.module._forward_hooks


def test_tdhook_activation_intervention_changes_output_and_removes_its_hook() -> None:
    policy = _policy()
    interaction = _interaction(policy)
    intervention = Intervention(
        "zero-head",
        InterventionTarget.ACTIVATION,
        InterventionTiming.OUTPUT,
        transform=torch.zeros_like,
        module_path="module",
    )
    adapter = TDHookInteractionAdapter(interaction)
    factory = TDHookInterventionFactory(interaction, (intervention,))

    with adapter.activate(factory) as active:
        result = active.invoke(interaction.representative_input.clone())
        assert torch.equal(result.get(("agents", "action")), torch.zeros(3, 2, 1))

    assert not policy.module._forward_hooks


def test_gradient_intervention_requires_an_autograd_enabled_interaction() -> None:
    interaction = _interaction(_policy())
    intervention = Intervention(
        "zero-gradient",
        InterventionTarget.GRADIENT,
        InterventionTiming.INPUT,
        transform=torch.zeros_like,
        module_path="module",
    )

    with pytest.raises(InterventionValidationError, match="gradient_enabled=True"):
        TDHookInterventionFactory(interaction, (intervention,))


@pytest.mark.parametrize("timing", [InterventionTiming.INPUT, InterventionTiming.OUTPUT])
def test_gradient_interventions_replace_backpropagated_input_gradients(timing: InterventionTiming) -> None:
    policy = _policy()
    interaction = _interaction(policy)
    interaction.descriptor = replace(interaction.descriptor, gradient_enabled=True)
    batch = interaction.representative_input.clone()
    observation = batch.get(("agents", "observation")).requires_grad_()
    batch.set(("agents", "observation"), observation)
    intervention = Intervention(
        f"zero-gradient-{timing.value}",
        InterventionTarget.GRADIENT,
        timing,
        transform=torch.zeros_like,
        module_path="module",
    )
    adapter = TDHookInteractionAdapter(interaction)
    factory = TDHookInterventionFactory(interaction, (intervention,))

    with adapter.activate(factory) as active:
        active.invoke(batch).get(("agents", "action")).sum().backward()

    assert observation.grad is not None
    assert torch.equal(observation.grad, torch.zeros_like(observation))


def test_intervention_paths_are_resolved_against_the_adapter_selection() -> None:
    interaction = _interaction(_policy())
    selected = TensorDictModule(
        torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "action")],
    )
    intervention = Intervention(
        "selected-head",
        InterventionTarget.ACTIVATION,
        InterventionTiming.OUTPUT,
        transform=torch.zeros_like,
        module_path="module.1",
    )
    adapter = TDHookInteractionAdapter(interaction, selected_module=selected)
    factory = TDHookInterventionFactory(interaction, (intervention,))

    with adapter.activate(factory) as active:
        result = active.invoke(interaction.representative_input.clone())
        assert torch.equal(result.get(("agents", "action")), torch.zeros(3, 2, 1))


class _TwoTensorOutputs(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value[..., :1], value[..., :1]


def test_tdhook_intervention_rejects_ambiguous_multi_tensor_outputs() -> None:
    policy = TensorDictModule(
        _TwoTensorOutputs(),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "action"), ("agents", "auxiliary")],
    )
    interaction = _interaction(policy)
    intervention = Intervention(
        "ambiguous",
        InterventionTarget.ACTIVATION,
        InterventionTiming.OUTPUT,
        transform=torch.zeros_like,
        module_path="module",
    )
    adapter = TDHookInteractionAdapter(interaction)
    factory = TDHookInterventionFactory(interaction, (intervention,))

    with adapter.activate(factory), pytest.raises(InterventionValidationError, match="2 tensor values"):
        adapter.invoke(interaction.representative_input.clone())
