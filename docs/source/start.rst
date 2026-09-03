Quickstart
==========

Install XDRL, interpret a native TorchRL policy, and capture one hidden
activation:

.. code-block:: bash

   pip install xdrl

.. code-block:: python

   import torch
   from tensordict import TensorDict
   from tensordict.nn import TensorDictModule
   from tdhook.latent import ActivationCaching
   from tdhook.workflow import Workflow
   from xdrl import interpret

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
   policy = interpret(policy)
   workflow = Workflow(
       ActivationCaching("module.1", cache_key=("activations", "hidden"))
   )
   execution = policy.run(workflow, batch)

   assert execution.data["action"].shape == (8, 2)
   assert execution.data["activations", "hidden", "module.1"].shape == (8, 8)

TorchRL and TensorDict own policy execution, keys, specs, parameters, and
batched data. TDHook owns the model-internal method: here, capturing the hidden
activation. XDRL's ``interpret`` view connects them and validates the call
boundary. An activation capture records an internal value; by itself, it is not
evidence that the activation causally affects the policy's action.

Native objectives
-----------------

TorchRL loss modules already encode an algorithm's components and functional
parameters. XDRL reads that structure directly:

.. code-block:: python

   objective = interpret(sac_loss)

   actor = objective.actor
   first_q = objective.qvalue[0]
   first_target_q = objective.target.qvalue[0]

Each component is directly callable and provides ``.run(workflow, data)``.
XDRL materializes the selected online, target, or ensemble-member parameters
only for that bounded call. DQN, PPO, SAC, IQL, and QMixer objectives are
supported explicitly; unsupported objectives fail instead of being inferred
from similarly named attributes.

Native ``ProbabilisticActor``, ``QValueActor``, ``ValueOperator``,
``ActorValueOperator``, and ``ActorCriticOperator`` objects expose the RL
functions they already define. A plain ``TensorDictModule`` becomes one
directly executable component, as in the quickstart above.

For recurrent TorchRL modules, see
:class:`xdrl.interpretation.RecurrentSemantics`. For richer model-internal
workflows, continue with :doc:`tutorials`.
