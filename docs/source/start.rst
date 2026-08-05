Getting Started
===============

``xdrl`` describes one TorchRL model invocation with native TensorDict keys,
TorchRL specs, and explicit execution semantics. TDHook can then observe or
intervene inside that declared interaction.

The following example is exercised by
``tests/integration/test_quickstart.py``:

.. code-block:: python

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
   batch = TensorDict(
       {"observation": torch.randn(8, 4)},
       batch_size=[8],
   )
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
   interaction = RuntimeInteractionContext(
       descriptor,
       policy,
       inputs,
       outputs,
       batch,
   )
   adapter = TDHookInteractionAdapter(
       interaction,
       aliases={"head": "module"},
   )

   with adapter.activate(ActivationCaching(r"module")) as active:
       result = active.invoke(batch.clone())
       activations = active.contexts[0].cache["module"]

   assert result["action"].shape == (8, 2)

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

What this guarantees
--------------------

The interaction validates required inputs and produced outputs, records its
evaluation lifecycle, switches the module to evaluation mode for the call,
and restores its previous state. The adapter validates TDHook paths and removes
all temporary hooks on normal or exceptional exit. Observation-only parity is
part of the declared conformance suite. The second call makes customisation
explicit: the intervention must preserve the selected activation's shape,
dtype, and device, and its hook is removed after the context exits.

What remains external
---------------------

TorchRL still owns environments, collectors, replay, losses, optimisation,
logging, and checkpointing. TDHook still owns activation capture, attribution,
probing, patching, and steering methods. Read :doc:`architecture` before using
recurrent or multi-agent contracts, and :doc:`compatibility` before changing
the pinned dependency revisions.
