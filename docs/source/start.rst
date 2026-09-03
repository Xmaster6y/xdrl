Quickstart
==========

Use a TDHook method on a TorchRL policy:

.. code-block:: bash

   pip install xdrl

.. code-block:: python

   import torch
   from tensordict import TensorDict
   from tensordict.nn import TensorDictModule
   from tdhook.latent import ActivationCaching
   from tdhook.workflow import Workflow
   from xdrl import interpret

   model = TensorDictModule(
       torch.nn.Sequential(
           torch.nn.Linear(4, 8),
           torch.nn.Tanh(),
           torch.nn.Linear(8, 2),
       ),
       in_keys=["observation"],
       out_keys=["action"],
   )
   policy = interpret(model)
   batch = TensorDict({"observation": torch.randn(8, 4)}, batch_size=[8])
   result = policy.run(
       Workflow(ActivationCaching("module.1", cache_key=("activations", "hidden"))),
       batch,
   )

   assert result.data["action"].shape == (8, 2)
   assert result.data["activations", "hidden", "module.1"].shape == (8, 8)

``interpret`` keeps the native TensorDict call and adds ``.run(...)`` for
TDHook workflows.

Select a network from a loss
----------------------------

TorchRL losses already hold their actor, critic, value, and target parameters.
XDRL exposes them directly:

.. code-block:: python

   sac = interpret(sac_loss)

   actor = sac.actor
   first_q = sac.qvalue[0]
   first_target_q = sac.target.qvalue[0]

Each selection is callable and provides ``.run(workflow, data)``. The last one
uses TorchRL's target parameters automatically; no second network description
is needed.

XDRL supports DQN, PPO, SAC, IQL, and QMixer losses, plus TorchRL actor, value,
Q-value, and actor-critic modules. Unknown losses fail clearly instead of
guessing their structure.

For recurrent TorchRL modules, see
:class:`xdrl.interpretation.RecurrentSemantics`. For richer model-internal
workflows, continue with :doc:`tutorials`.
