Getting Started
===============

``xdrl`` describes one TorchRL model interaction with native TensorDict keys,
TorchRL specs, and explicit RL execution semantics. TDHook v0.2 owns configured
methods, targets, hooks, workflows, planning, and TensorDict artifacts; XDRL
validates every actual root model call made by that workflow.

The following example is exercised by
``tests/integration/test_quickstart.py``:

.. code-block:: python

   import torch
   from tensordict import TensorDict
   from tensordict.nn import TensorDictModule
   from tdhook.latent import ActivationCaching
   from tdhook.session import HookSession
   from tdhook.targets import Target
   from tdhook.workflow import Workflow

   from xdrl import (
       BatchSemantics,
       InteractionContract,
       InteractionPhase,
       KeyPresence,
       KeyRole,
       KeySchema,
       ModelRole,
       RuntimeInteractionContext,
       TDHookWorkflowRunner,
       TensorDictSchema,
   )

   inputs = TensorDictSchema(
       (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),),
       BatchSemantics(("env",)),
   )
   outputs = TensorDictSchema(
       (KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),),
       BatchSemantics(("env",)),
   )
   policy = TensorDictModule(
       torch.nn.Linear(4, 2),
       in_keys=["observation"],
       out_keys=["action"],
   )
   batch = TensorDict({"observation": torch.randn(8, 4)}, batch_size=[8])
   contract = InteractionContract(
       identity="policy:evaluation:0",
       role=ModelRole.ACTOR,
       phase=InteractionPhase.EVALUATION,
       module_path="policy",
       input_schema=inputs,
       output_schema=outputs,
       module_training=False,
   )
   interaction = RuntimeInteractionContext(contract, policy, batch)

   workflow = Workflow(
       ActivationCaching("module", cache_key=("activations", "head"))
   )
   runner = TDHookWorkflowRunner(interaction)
   plan = runner.plan(workflow, batch.clone())
   execution = runner.run(
       workflow,
       batch.clone(),
       code_revision="example-revision",
       expected_plan=plan,
   )

   assert execution.data["action"].shape == (8, 2)
   assert "module" in execution.data["activations", "head"]
   assert execution.provenance.model_calls == plan.model_passes == 1

   target = Target("module", "activation", -1, (0, 1))
   with interaction, HookSession(policy) as session:
       session.replace(target, 0)
       intervened = interaction.invoke(batch.clone())

   assert torch.equal(intervened["action"], torch.zeros(8, 2))
   assert not policy.module._forward_hooks

What this guarantees
--------------------

TDHook determines the workflow dependencies, compatibility groups, and pass
count. XDRL keeps the original model identity and target paths, applies the RL
execution policy, validates every root TensorDict model call, records its
lifecycle, and rejects disagreement between planned and observed calls. Both
libraries restore their temporary state after normal or exceptional exits.

What remains external
---------------------

TorchRL owns environments, collectors, replay, losses, and optimisation.
TDHook owns model-internal methods and interactive target operations. The
experiment owns the meaning of an intervention, scientific controls, scoring,
and conclusions. Read :doc:`architecture` for recurrent and multi-agent
contracts and :doc:`compatibility` before changing the locked dependencies.
