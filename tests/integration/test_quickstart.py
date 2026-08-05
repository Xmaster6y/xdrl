import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent.activation_caching import ActivationCaching

from xdrl import (
    BatchSemantics,
    InteractionDescriptor,
    InteractionPhase,
    Intervention,
    InterventionTarget,
    InterventionTiming,
    KeyPresence,
    KeyRole,
    KeySchema,
    ModelRole,
    RuntimeInteractionContext,
    SchemaSnapshot,
    TDHookInteractionAdapter,
    TDHookInterventionFactory,
    TensorDictSchema,
)


def test_documented_quickstart_observes_a_typed_policy_interaction() -> None:
    inputs = TensorDictSchema(
        (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),),
        BatchSemantics(("env",)),
    )
    outputs = TensorDictSchema(
        (KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),),
        BatchSemantics(("env",)),
    )
    policy = TensorDictModule(torch.nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"])
    batch = TensorDict({"observation": torch.randn(8, 4)}, batch_size=[8])
    descriptor = InteractionDescriptor(
        identity="policy:evaluation:0",
        role=ModelRole.ACTOR,
        phase=InteractionPhase.EVALUATION,
        module_path="policy",
        input_schema=SchemaSnapshot.from_schema(inputs),
        output_schema=SchemaSnapshot.from_schema(outputs),
        batch_dimensions=("env",),
        module_training=False,
    )
    interaction = RuntimeInteractionContext(descriptor, policy, inputs, outputs, batch)
    adapter = TDHookInteractionAdapter(interaction, aliases={"head": "module"})

    with adapter.activate(ActivationCaching(r"module")) as active:
        result = active.invoke(batch.clone())
        cache = active.contexts[0].cache

    assert result["action"].shape == (8, 2)
    assert "module" in cache
    assert adapter.target_paths == {"head": "td_module.module"}
    assert not policy.module._forward_hooks

    intervention = Intervention(
        "zero-policy-head",
        InterventionTarget.ACTIVATION,
        InterventionTiming.OUTPUT,
        transform=torch.zeros_like,
        module_path="module",
    )
    factory = TDHookInterventionFactory(interaction, (intervention,))
    with adapter.activate(factory) as active:
        intervened = active.invoke(batch.clone())

    assert torch.equal(intervened["action"], torch.zeros(8, 2))
    assert not policy.module._forward_hooks
