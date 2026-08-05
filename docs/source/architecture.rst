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
``module_training`` optionally requests train or evaluation mode for the live
call; the exact pre-existing mode of every submodule is restored afterwards.

``RuntimeInteractionContext`` is the matching ephemeral wrapper. Its
construction checks a representative input against the declared input schema;
``invoke`` checks live inputs and outputs around the actual module call. The
context restores exploration, gradient/inference, autocast, and a supplied
hook context even if the invocation raises. Its ordered ``before``, ``after``,
and ``failure`` records retain only phase, module path, error text, and key
shapes. An interaction identity must be stable within an execution record;
event order is increasing within that identity.

Execution-boundary support
--------------------------

The context itself is a one-shot callable, so a direct call and a local
``SyncDataCollector`` policy can use the same object. It does not own a
collector, rollout, replay buffer, optimiser, or trainer schedule.

.. list-table:: Support matrix
   :header-rows: 1

   * - Boundary
     - Typical model role
     - TensorDict available
     - Gradients available
     - Lifecycle guarantee
   * - Direct call
     - Any
     - Caller input/output
     - Descriptor-controlled
     - One before/after or failure pair per call
   * - Synchronous collection
     - Actor
     - Current env step
     - Normally disabled
     - Main-process policy calls only
   * - Evaluation rollout
     - Actor/value
     - Current env step
     - Normally disabled
     - Exploration and exact module modes restored
   * - Replay-batch loss
     - Loss/critic/value
     - Sampled replay batch
     - Usually enabled
     - Context covers the loss-module call
   * - Target/value estimate
     - Value/critic
     - Loss working batch
     - Usually disabled
     - Separate target interaction identity
   * - Backward/optimisation
     - Loss
     - Loss output and gradients
     - Enabled when requested
     - Hooks live only while the context remains open

For backward-time gradient observation, use the explicit context form and call
``backward`` before leaving it. The one-shot callable necessarily removes
temporary hooks after the forward call returns. Optimiser stepping remains a
trainer responsibility and does not create a model invocation by itself.

Only local synchronous calls are supported. Hooks installed in the main
process are not propagated to worker copies. Multiprocessing and distributed
collectors, compiled modules, and CUDA graphs are unsupported until their
copying and lifecycle semantics are tested explicitly.

Recurrent and multi-agent semantics
-----------------------------------

``RecurrentSemantics`` maps each current recurrent-state key to its produced
``next`` key, names boolean reset masks, and records the time axis, burn-in,
truncated window, and collector mode. Direct calls, synchronous collection,
and replay sequences are the supported state lifecycles. Multiprocess,
asynchronous, and distributed recurrent collectors fail at contract creation
instead of implying that state and hooks propagate to worker copies.

``MultiAgentSemantics`` separates implementation paths from semantic targets.
It records independent, parameter-shared, centralised-critic, or mixer
topology together with the TorchRL group, agent count, and a
``SemanticTarget`` made from a model role and ``AgentSelector``. In
particular, a shared policy path cannot stand in for a per-agent identity:
targeting one or more agents is expressed only by the selector. Centralised
critics use critic/value roles, and mixers use the mixer role. VMAS-style
``("agents", ...)`` keys remain native TensorDict nested keys; the agent axis
is named separately from environment and time batch axes.

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
graphs. Sampling is opt-in. Each ``DimensionReduction`` pairs an explicit
batch-dimension name (such as ``time`` or ``agent``) with a serialised
``mean``, ``sum``, or ``max`` policy; dimensions are preserved by default.
``max_records`` plus an overflow policy bounds memory, while an optional
callback supports streaming consumers. Probes and attribution remain external
consumers of these records rather than becoming xdrl algorithms.

TDHook v0.2 workflows
---------------------

``TDHookWorkflowRunner`` delegates composition, planning, coexecution,
TensorDict artifacts, target resolution, and hook cleanup to TDHook's public
``Workflow`` API. XDRL installs temporary public hooks on the original root
TensorDict module so every actual model pass crosses the live RL input/output
schema and execution-mode boundary without wrapping the model, changing its
class, or shifting TDHook target paths.

``runner.plan(workflow, data)`` returns TDHook's immutable ``WorkflowPlan``.
``runner.run(...)`` returns the native final TensorDict together with a
tensor-free ``WorkflowRunRecord``. The runner rejects plan drift, incompatible
gradient requirements, and a disagreement between TDHook's declared model-pass
count and the model calls XDRL actually observed. Lazy modules must be
materialised explicitly. Compiled, distributed, and remote modules remain
unsupported.

Interactive capture or replacement uses TDHook's ``HookSession`` directly
inside an active XDRL interaction. XDRL does not implement a second hook,
target, or intervention runtime.
