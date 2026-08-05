import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.session import HookSession
from tdhook.targets import Target
from tdhook.workflow import Workflow

from xdrl import (
    BatchSemantics,
    InteractionDescriptor,
    InteractionPhase,
    KeyPresence,
    KeyRole,
    KeySchema,
    ModelRole,
    RuntimeInteractionContext,
    SchemaSnapshot,
    TDHookWorkflowRunner,
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
    workflow = Workflow(ActivationCaching("module", cache_key=("activations", "head")))
    execution = TDHookWorkflowRunner(interaction).run(workflow, batch.clone())
    result = execution.data
    cache = result["activations", "head"]

    assert result["action"].shape == (8, 2)
    assert "module" in cache
    assert execution.record.model_calls == 1
    assert not policy.module._forward_hooks

    target = Target("module", "activation", -1, (0, 1))
    with interaction, HookSession(policy) as session:
        session.replace(target, 0)
        intervened = interaction.invoke(batch.clone())

    assert torch.equal(intervened["action"], torch.zeros(8, 2))
    assert not policy.module._forward_hooks
