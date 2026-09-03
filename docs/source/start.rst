Quickstart
==========

Install XDRL and construct one boundary around an unchanged TorchRL module:

.. code-block:: bash

   pip install xdrl

.. code-block:: python

   import torch
   from tensordict import TensorDict
   from tensordict.nn import TensorDictModule
   from xdrl import Interaction

   policy = TensorDictModule(
       torch.nn.Linear(4, 2),
       in_keys=["observation"],
       out_keys=["action"],
   )
   interaction = Interaction(policy)
   batch = TensorDict(
       {"observation": torch.randn(8, 4)},
       batch_size=[8],
       names=["env"],
   )

   result = interaction(batch)
   assert result["action"].shape == (8, 2)

The TorchRL module remains the source of truth for keys and specs. XDRL only
adds semantic checks that the upstream objects do not express. For example,
TorchRL recurrent modules can use their native ``next`` and ``is_init``
conventions directly:

.. code-block:: python

   from torchrl.modules import LSTMModule
   from xdrl import RecurrentSemantics

   recurrent_policy = LSTMModule(
       input_size=4,
       hidden_size=8,
       in_key="observation",
       out_key="embedding",
   )
   recurrent = RecurrentSemantics.from_torchrl(
       "recurrent_state_h",
       "recurrent_state_c",
   )
   recurrent_interaction = Interaction(
       recurrent_policy,
       recurrent,
   )

TDHook workflows
----------------

``run_workflow`` is the only XDRL workflow entrypoint. It validates the
caller's Torch state against TDHook's public plan, delegates execution to
TDHook, and returns TDHook's native ``WorkflowResult``:

.. code-block:: python

   from tdhook.latent import ActivationCaching
   from tdhook.workflow import Workflow
   from xdrl import run_workflow

   workflow = Workflow(ActivationCaching("module"))
   execution = run_workflow(interaction, workflow, batch.clone())

   assert execution.plan.model_passes == 1

Use TDHook ``Target`` and ``HookSession`` directly for model-internal captures,
replacements, gradients, and repeated-call occurrence selection.
