Quickstart
==========

.. code-block:: bash

   pip install xdrl

.. code-block:: python

   import torch
   from tensordict import TensorDict
   from tensordict.nn import TensorDictModule
   from xdrl import (
       BatchSemantics,
       KeyPresence,
       KeyRole,
       KeySchema,
       ModelRole,
       TensorDictSchema,
       validate_module,
   )

   batch = TensorDict({"observation": torch.randn(8, 4)}, batch_size=[8])
   policy = TensorDictModule(
       torch.nn.Linear(4, 2),
       in_keys=["observation"],
       out_keys=["action"],
   )
   policy.role = ModelRole.ACTOR
   batch_dims = BatchSemantics(("env",))
   policy.input_schema = TensorDictSchema(
       (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),),
       batch_dims,
   )
   policy.output_schema = TensorDictSchema(
       (KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),),
       batch_dims,
   )

   result = validate_module(policy, batch)
   assert result["action"].shape == (8, 2)
