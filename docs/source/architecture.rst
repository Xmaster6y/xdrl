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

Interaction contexts
--------------------

``xdrl.interactions`` records an individual invocation of a TensorDict module
without changing TorchRL's collector or trainer loops. An
``InteractionDescriptor`` is the durable, serialisable record: it identifies
the role, phase, module path, declared I/O schemas, batch semantics,
exploration/gradient/autocast configuration, and any supplied logical
step/episode/trajectory identifiers. It contains neither tensors nor modules.

``RuntimeInteractionContext`` is the matching ephemeral wrapper. Its
construction checks a representative input against the declared input schema;
``invoke`` checks live inputs and outputs around the actual module call. The
context restores exploration, gradient/inference, autocast, and a supplied
hook context even if the invocation raises. Its ordered ``before``, ``after``,
and ``failure`` records retain only phase, module path, error text, and key
shapes. An interaction identity must be stable within an execution record;
event order is increasing within that identity.

Observation traces
------------------

``xdrl.observations`` adds bounded typed records around an interaction without
changing its TensorDict invocation. ``ObservationTrace`` captures module input
and output keys automatically when supplied to ``RuntimeInteractionContext``;
hook users can record activations or gradients directly with
``observe_tensor``. Each record includes interaction identity, phase, target,
hook direction, nested key path, batch/time/agent semantics, and
model/checkpoint identity and exploration metadata, but ``to_dict`` always excludes
the optional tensor payload.

``RetentionPolicy`` makes retention explicit: metadata-only is the default;
detached or CPU snapshots clone their values and never retain computation
graphs. Sampling and named ``mean``, ``sum``, or ``max`` batch reductions are
opt-in. ``max_records`` plus an overflow policy bounds memory, while an
optional callback supports streaming consumers. Probes and attribution remain
external consumers of these records rather than becoming xdrl algorithms.

TDHook instrumentation
----------------------

``TDHookInteractionAdapter`` binds one ``RuntimeInteractionContext`` to one or
more raw TDHook context factories. It validates that the selected TensorDict
input and output keys satisfy both the model and interaction contract, then
executes through the existing context without changing TensorDict nesting,
batch shape, dtype, device, or model mode. Semantic aliases in the interaction
descriptor are exposed as stable TDHook target paths.

Lazy modules must be explicitly materialised with ``adapter.materialize()``
before ``adapter.activate(factory)``. Compiled, distributed, and remote modules
are rejected instead of being instrumented with ambiguous semantics.
