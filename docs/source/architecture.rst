Architecture: TorchRL-native model contracts
============================================

``xdrl.types`` describes RL model roles and TensorDict I/O contracts without
introducing a second data container or a parallel specification hierarchy.

Mapping to TorchRL and TensorDict
---------------------------------

=============================  ===============================================
xdrl type                      Reused or refined upstream type
=============================  ===============================================
``TensorDictKey``              ``tensordict.utils.NestedKey``
``TensorDictSchema`` input     ``tensordict.TensorDictBase``
``KeySchema.spec``             ``torchrl.data.TensorSpec`` (including Composite)
``TorchRLModule``              ``tensordict.nn.TensorDictModuleBase``
``ContractModule``             Structural interface for TorchRL-compatible modules
=============================  ===============================================

``ModelRole`` makes actor, critic, value, loss, encoder, mixer, and world-model
intent explicit. ``KeyRole`` records state, observation, action, reward,
termination, log-probability, distribution-parameter, value, and feature keys.
All key paths remain native nested keys, for example
``("agents", "action")``.

Validation boundaries
---------------------

Static annotations specify module roles, nested keys, and the
``ContractModule`` interface. Construction-time checks reject duplicate key
paths. Runtime ``validate_module`` checks a required input schema before a
call, then produced keys afterwards; errors include the nested key path and
the observed shape. ``BatchSemantics`` names leading TensorDict dimensions
separately from the feature shape held by a ``TensorSpec``.

Use runtime validation at public composition boundaries, not in inner training
loops. TorchRL's probabilistic actors, value operators, loss modules, and
exploration modules continue to own their native construction and behavior;
the contract is only an optional annotation and validation boundary.
