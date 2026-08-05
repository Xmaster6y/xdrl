# XDRL 0.2 public API

Use this reference for `xdrl==0.2.*`. All XDRL symbols below come from the public package namespace.

## Ownership and support

TensorDict and TorchRL own data, modules, collectors, environments, replay, losses, specs, and exploration. TDHook 0.2 owns workflows, targets, hook programs, model-internal access, artifacts, cleanup, and pass counts. XDRL owns typed RL interaction semantics, boundary validation, execution-state restoration, lifecycle evidence, compatibility, and provenance.

Supported execution is local, synchronous, and eager. Compiled, remote, distributed, multiprocessing, asynchronous-collector, CUDA-graph, and worker-copied hook paths are unsupported unless a later conformance contract says otherwise.

Installation alone is not compatibility evidence. Call `validate_runtime_compatibility()` and name the relevant conformance suite before claiming support. `BatchSemantics` maps semantic labels to positional leading TensorDict batch axes; it does not add named axes to TensorDict.

## Typed observation

<!-- runnable-example -->
```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from xdrl import BatchSemantics, InteractionContract, InteractionPhase, KeyPresence
from xdrl import KeyRole, KeySchema, ModelRole, ObservationTrace
from xdrl import RuntimeInteractionContext, TensorDictSchema

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
contract = InteractionContract(
    "actor:evaluation:0", ModelRole.ACTOR, InteractionPhase.EVALUATION, "policy",
    inputs, outputs,
    module_training=False,
)
trace = ObservationTrace()
interaction = RuntimeInteractionContext(contract, policy, batch, observations=trace)
result = interaction(batch.clone())

assert result["action"].shape == (8, 2)
assert trace.records and all(record.payload is None for record in trace.records)
```

Metadata-only retention is the safe default. Output-parity evidence lives at `tests/unit/test_observations.py::test_trace_is_serialisable_and_observation_only_preserves_model_output`.

## TensorDict-boundary intervention

<!-- runnable-example -->
```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from xdrl import BatchSemantics, InteractionContract, InteractionPhase, Intervention
from xdrl import InterventionController, InterventionTarget, InterventionTiming
from xdrl import KeyPresence, KeyRole, KeySchema, ModelRole, RuntimeInteractionContext
from xdrl import TensorDictSchema, run_paired

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
contract = InteractionContract(
    "actor:evaluation:paired", ModelRole.ACTOR, InteractionPhase.EVALUATION, "policy",
    inputs, outputs,
    checkpoint_id="checkpoint-1",
)
baseline = RuntimeInteractionContext(contract, policy, batch)
edit = Intervention(
    "add-one", InterventionTarget.TENSORDICT, InterventionTiming.OUTPUT,
    transform=lambda value: value + 1, key="action",
)
steered = RuntimeInteractionContext(
    contract, policy, batch,
    interventions=InterventionController((edit,)),
)
pair = run_paired(baseline, steered, batch)

assert torch.equal(pair.intervention["action"], pair.baseline["action"] + 1)
```

XDRL interventions edit declared TensorDict input or output keys. Use TDHook `Target` and `HookSession` for activation, gradient, or parameter interventions. Matched mechanics and provenance do not establish a causal conclusion.

## TDHook 0.2 workflow execution

<!-- runnable-example -->
```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow
from xdrl import BatchSemantics, InteractionContract, InteractionPhase, KeyPresence
from xdrl import KeyRole, KeySchema, ModelRole, RuntimeInteractionContext
from xdrl import TDHookWorkflowRunner, TensorDictSchema

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
contract = InteractionContract(
    "actor:evaluation:workflow", ModelRole.ACTOR, InteractionPhase.EVALUATION, "policy",
    inputs, outputs,
    module_training=False,
)
interaction = RuntimeInteractionContext(contract, policy, batch)
workflow = Workflow(ActivationCaching("module", cache_key=("activations", "head")))
runner = TDHookWorkflowRunner(interaction)
plan = runner.plan(workflow, batch.clone())
execution = runner.run(
    workflow, batch.clone(), code_revision="example-revision", expected_plan=plan
)

assert execution.provenance.model_calls == plan.model_passes == 1
assert "module" in execution.data["activations", "head"]
```

`TDHookWorkflowRunner` delegates planning and method execution to TDHook. XDRL observes successful root calls through public PyTorch hooks, validates each interaction boundary, restores execution state, and checks the observed count against TDHook's plan. `WorkflowProvenance` records public plan-to-call evidence, dependency versions, code revision, and an optional seed; it is not TDHook artifact provenance or a scientific result.

## Evidence map

- `tests/unit`: schemas, interactions, observations, interventions, recurrent and multi-agent contracts, and workflow checks.
- `tests/integration`: XDRL, TorchRL, and TDHook composition and plan-to-call accounting.
- `tests/upstream_compatibility`: version-sensitive upstream boundaries.
- `tests/behavioural_parity`: native versus observation-only instrumented output parity.

State install/import, runtime compatibility, named conformance results, supported execution mode, and behavioral conclusions separately.
