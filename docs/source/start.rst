Quickstart
==========

Install XDRL, run a small TorchRL policy, and capture one hidden activation:

.. code-block:: bash

   pip install xdrl

.. code-block:: python

   import torch
   from tensordict import TensorDict
   from tensordict.nn import TensorDictModule
   from tdhook.latent import ActivationCaching
   from tdhook.workflow import Workflow
   from xdrl import Interaction, run_workflow

   policy = TensorDictModule(
       torch.nn.Sequential(
           torch.nn.Linear(4, 8),
           torch.nn.Tanh(),
           torch.nn.Linear(8, 2),
       ),
       in_keys=["observation"],
       out_keys=["action"],
   )
   batch = TensorDict(
       {"observation": torch.randn(8, 4)},
       batch_size=[8],
   )
   interaction = Interaction(policy)
   workflow = Workflow(
       ActivationCaching("module.1", cache_key=("activations", "hidden"))
   )
   execution = run_workflow(interaction, workflow, batch)

   assert execution.data["action"].shape == (8, 2)
   assert execution.data["activations", "hidden", "module.1"].shape == (8, 8)

TorchRL and TensorDict own policy execution, keys, specs, and batched data.
TDHook owns the model-internal method: here, capturing the hidden activation.
XDRL validates those boundaries and connects the policy to the workflow through
``Interaction`` and ``run_workflow``. An activation capture records an internal
value; by itself, it is not evidence that the activation causally affects the
policy's action.

For recurrent TorchRL modules, see
:class:`xdrl.interactions.RecurrentSemantics`. For richer model-internal
workflows, continue with :doc:`tutorials`.
