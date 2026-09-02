Quickstart
==========

Install XDRL and construct one typed boundary around an unchanged TorchRL
module:

.. code-block:: bash

   pip install xdrl

.. code-block:: python

   import torch
   from tensordict import TensorDict
   from tensordict.nn import TensorDictModule
   from xdrl import (
       BatchSemantics,
       Interaction,
       InteractionSpec,
       KeyRole,
       KeySchema,
       ModelRole,
       TensorDictSchema,
   )

   policy = TensorDictModule(
       torch.nn.Linear(4, 2),
       in_keys=["observation"],
       out_keys=["action"],
   )
   interaction = Interaction(
       policy,
       InteractionSpec(
           ModelRole.ACTOR,
           TensorDictSchema((KeySchema("observation", KeyRole.OBSERVATION),)),
           TensorDictSchema((KeySchema("action", KeyRole.ACTION),)),
           BatchSemantics(("env",)),
       ),
   )
   batch = TensorDict({"observation": torch.randn(8, 4)}, batch_size=[8])

   result = interaction(batch)
   assert result["action"].shape == (8, 2)

TDHook workflows
----------------

``run_workflow`` is the only XDRL workflow entrypoint. It returns TDHook's
native ``WorkflowResult`` without re-planning or reproducing hook state:

.. code-block:: python

   from tdhook.latent import ActivationCaching
   from tdhook.workflow import Workflow
   from xdrl import run_workflow

   workflow = Workflow(ActivationCaching("module"))
   execution = run_workflow(interaction, workflow, batch.clone())

   assert execution.plan.model_passes == 1

Use TDHook ``Target`` and ``HookSession`` directly for model-internal captures,
replacements, gradients, and repeated-call occurrence selection.
