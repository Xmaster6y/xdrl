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
TDHook workflows. The policy, TensorDict keys, and execution behavior remain
native to TorchRL; XDRL only supplies the interpretability view.

For recurrent TorchRL modules, see
:class:`xdrl.interpretation.RecurrentSemantics`. For richer model-internal
workflows, continue with :doc:`tutorials`.
