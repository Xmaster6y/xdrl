# XDRL 0.1 public API

Use this reference for `xdrl==0.1.*`. All XDRL symbols in the examples come from the public package namespace.

## Ownership and support

| Concern | Owner |
| --- | --- |
| TensorDict data, nested keys, tensor specs, modules, collectors, environments, replay, losses, and exploration | TensorDict / TorchRL |
| Hooks, activation and gradient access, attribution, probing, patching, steering, pipeline planning, artifacts, and pass counts | TDHook |
| Typed RL interaction semantics, validation, lifecycle restoration, binding to TDHook, compatibility, and provenance | XDRL |

Supported execution is local and synchronous: direct TensorDict module calls, synchronous collector policies, deterministic evaluation, replay/loss calls, targets, and optimisation/backward contexts. Recurrent execution is limited to direct, synchronous, and replay-sequence state lifecycles. Compiled, remote, distributed, multiprocessing, asynchronous-collector, CUDA-graph, and worker-copied hook paths are unsupported unless a later version names new conformance evidence.

Installation alone is not compatibility evidence. Call `validate_runtime_compatibility()` and name the relevant conformance suite before claiming support.

## Typed observation

<!-- runnable-example -->
```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from xdrl import (
    BatchSemantics,
    InteractionDescriptor,
    InteractionPhase,
    KeyPresence,
    KeyRole,
    KeySchema,
    ModelRole,
    ObservationTrace,
    RuntimeInteractionContext,
    SchemaSnapshot,
    TensorDictSchema,
)

inputs = TensorDictSchema(
    (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),),
    BatchSemantics(("env",)),
)
outputs = TensorDictSchema(
    (KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),),
    BatchSemantics(("env",)),
)
policy = TensorDictModule(torch.nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"])
batch = TensorDict({"observation": torch.randn(8, 4)}, batch_size=[8])
descriptor = InteractionDescriptor(
    identity="actor:evaluation:0",
    role=ModelRole.ACTOR,
    phase=InteractionPhase.EVALUATION,
    module_path="policy",
    input_schema=SchemaSnapshot.from_schema(inputs),
    output_schema=SchemaSnapshot.from_schema(outputs),
    batch_dimensions=("env",),
    model_id="actor-v1",
    checkpoint_id="sha256:example",
    exploration_mode="deterministic",
    module_training=False,
)
trace = ObservationTrace()  # metadata-only retention is the safe default
interaction = RuntimeInteractionContext(descriptor, policy, inputs, outputs, batch, observations=trace)
result = interaction(batch.clone())

assert result["action"].shape == (8, 2)
assert trace.records and all(record.payload is None for record in trace.records)
```

For opt-in payloads, construct `RetentionPolicy` with `TensorRetention.DETACHED` or `CPU`, named `DimensionReduction` operations, and a bounded record policy. The metadata-only trace flow above is exercised by `tests/unit/test_observations.py::test_trace_is_serialisable_and_observation_only_preserves_model_output`; TDHook adapter parity is covered separately by `tests/behavioural_parity/test_tdhook_adapter.py`. Tensor retention is not a license to keep unbounded rollout data.

## Checked intervention and paired execution

<!-- runnable-example -->
```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from xdrl import (
    BatchSemantics,
    InteractionDescriptor,
    InteractionPhase,
    Intervention,
    InterventionController,
    InterventionTarget,
    InterventionTiming,
    KeyPresence,
    KeyRole,
    KeySchema,
    ModelRole,
    RuntimeInteractionContext,
    SchemaSnapshot,
    TensorDictSchema,
    run_paired,
)

inputs = TensorDictSchema(
    (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),),
    BatchSemantics(("env",)),
)
outputs = TensorDictSchema(
    (KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),),
    BatchSemantics(("env",)),
)
policy = TensorDictModule(torch.nn.Linear(2, 1, bias=False), in_keys=["observation"], out_keys=["action"])
batch = TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2])
descriptor = InteractionDescriptor(
    "actor:evaluation:paired",
    ModelRole.ACTOR,
    InteractionPhase.EVALUATION,
    "policy",
    SchemaSnapshot.from_schema(inputs),
    SchemaSnapshot.from_schema(outputs),
    batch_dimensions=("env",),
    checkpoint_id="checkpoint-1",
)
baseline = RuntimeInteractionContext(descriptor, policy, inputs, outputs, batch)
edit = Intervention(
    "add-one",
    InterventionTarget.TENSORDICT,
    InterventionTiming.OUTPUT,
    transform=lambda value: value + 1,
    key="action",
)
steered = RuntimeInteractionContext(
    descriptor,
    policy,
    inputs,
    outputs,
    batch,
    interventions=InterventionController((edit,)),
)
pair = run_paired(baseline, steered, batch)

assert torch.equal(pair.intervention["action"], pair.baseline["action"] + 1)
assert pair.checkpoint_id == "checkpoint-1"
```

Use `TDHookInterventionFactory` for activation or gradient targets, then pass it to `TDHookInteractionAdapter.activate`. A transform must preserve shape, dtype, and device. Paired execution provides matched mechanics and provenance, not a causal conclusion.

## Recurrent and multi-agent declarations

<!-- runnable-example -->
```python
import json

from xdrl import (
    AgentSelector,
    BatchSemantics,
    InteractionDescriptor,
    InteractionPhase,
    InteractionTopology,
    KeyPresence,
    KeyRole,
    KeySchema,
    ModelRole,
    MultiAgentSemantics,
    RecurrentCollectorMode,
    RecurrentSemantics,
    RecurrentStateTransition,
    SchemaSnapshot,
    SemanticTarget,
    TensorDictSchema,
)

inputs = TensorDictSchema(
    (
        KeySchema("state", KeyRole.STATE, KeyPresence.REQUIRED),
        KeySchema("is_init", KeyRole.TERMINATION, KeyPresence.REQUIRED),
    ),
    BatchSemantics(("env", "time")),
)
outputs = TensorDictSchema(
    (KeySchema(("next", "state"), KeyRole.STATE, KeyPresence.PRODUCED),),
    BatchSemantics(("env", "time")),
)
recurrent = RecurrentSemantics(
    transitions=(RecurrentStateTransition(("state",), ("next", "state")),),
    reset_keys=(("is_init",),),
    sequence_dimension="time",
    burn_in=2,
    truncated_window=8,
    collector_mode=RecurrentCollectorMode.REPLAY_SEQUENCE,
)
multi_agent = MultiAgentSemantics(
    topology=InteractionTopology.PARAMETER_SHARED,
    group="agents",
    n_agents=3,
    target=SemanticTarget(ModelRole.ACTOR, AgentSelector("agents")),
)
descriptor = InteractionDescriptor(
    "actor:replay:0",
    ModelRole.ACTOR,
    InteractionPhase.REPLAY,
    "policy.rnn",
    SchemaSnapshot.from_schema(inputs),
    SchemaSnapshot.from_schema(outputs),
    batch_dimensions=("env", "time"),
    time_dimension="time",
    agent_dimension="agent",
    recurrent=recurrent,
    multi_agent=multi_agent,
)
encoded = json.loads(json.dumps(descriptor.to_dict()))

assert encoded["recurrent"]["collector_mode"] == "replay_sequence"
assert encoded["multi_agent"]["target"]["selector"]["group"] == "agents"
```

Required recurrent state must map to produced next state, reset masks must be declared inputs, and the sequence dimension must match the descriptor. Multi-agent targeting uses group, agent selection, role, and topology rather than Python module identity.

## TDHook binding, plans, and provenance

Use `TDHookInteractionAdapter(interaction, aliases={...})` to resolve stable semantic aliases to TDHook paths. For one or more raw factories, enter `with adapter.activate(factory) as active:` and call `active.invoke(batch)`. Materialise lazy modules explicitly with `adapter.materialize()` before activation.

For a TDHook `Pipeline`, call `adapter.run_pipeline(pipeline, artifacts, code_revision=..., seed=...)`. TDHook remains the owner of planning, stage grouping, artifacts, and pass count; XDRL validates every planned model call and returns `TDHookPipelineResult` with per-stage `interaction_provenance`.

For direct factories, capture a `ProvenanceManifest` with the descriptor, selected keys, `adapter.target_paths`, serialisable TDHook method configuration, exact dependency versions, and code revision. `to_json()` is deterministic. Provenance establishes reproducibility metadata; it does not establish support or scientific validity.

## Evidence map

- `tests/unit`: local schema, interaction, observation, intervention, recurrent, and multi-agent contracts.
- `tests/integration`: XDRL/TorchRL/TDHook composition, planned workflows, and provenance.
- `tests/upstream_compatibility`: version-sensitive and private upstream boundaries.
- `tests/behavioural_parity`: native versus observation-only instrumented output parity.

State install/import, runtime compatibility, named conformance results, supported execution mode, and behavioral conclusions separately.
