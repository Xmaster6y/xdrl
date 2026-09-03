---
name: xdrl-library
description: Build, review, or explain XDRL interpretability extensions for native TorchRL modules and objectives.
---

# XDRL library

Use XDRL to discover the RL components and parameterizations already encoded by
native TorchRL modules and objectives, then pass the selected component to
TDHook.

## Ownership

- TensorDict and TorchRL own data, modules, specs, collectors, environments,
  replay, losses, and optimisation.
- TDHook owns model-internal targets, hooks, captures, replacements, workflows,
  planning, occurrences, artifacts, and cleanup.
- XDRL owns explicit TorchRL adapters, bounded functional-parameter binding,
  the TDHook bridge, and minimal recurrent boundary checks.
- Applications own experiment pairing, artifact metadata, reproducibility
  manifests, and scientific interpretation.

Do not introduce an XDRL hook implementation, configuration model of the RL
system, provenance format, trainer, data container, or paired-experiment
subsystem.

## Entry points

1. Call `interpret(module)` or `interpret(loss)` on the existing TorchRL object.
2. Select the actor, critic, value, Q-value member, mixer, or target component
   already exposed by that object.
3. Call the component directly or call `component.run(workflow, data)`.
4. Use TDHook `Target(occurrences=(...,))` or `HookSession` directly for repeated
   model-internal calls.

TensorDict's `batch_size` and dimension names describe the batch. The module's
`in_keys` and `out_keys` are the boundary. Use TorchRL `SafeModule` when values
need native spec enforcement. Use
`RecurrentSemantics.from_torchrl(...)` for TorchRL's `next` and `is_init`
conventions. The caller owns training, autograd, inference, exploration, and
autocast contexts; `Component.run` checks autograd against TDHook's declared
requirements before execution.

`Component.run` returns TDHook's native `WorkflowResult`.

<!-- runnable-example -->
```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow
from xdrl import interpret

policy = TensorDictModule(torch.nn.Linear(4, 2), ["observation"], ["action"])
policy = interpret(policy)
data = TensorDict({"observation": torch.randn(8, 4)}, batch_size=[8])

native = policy(data.clone())
result = policy.run(
    Workflow(ActivationCaching("module", cache_key=("activations", "policy"))),
    data.clone(),
)

assert native["action"].shape == (8, 2)
assert result.plan.model_passes == 1
```

For a supported TorchRL loss, never redeclare its architecture. Use the
algorithm adapter directly, for example `interpret(sac_loss).target.qvalue[0]`.
Explicit adapters exist for DQN, PPO, SAC, IQL, and QMixer. Unsupported losses
must fail closed or register a dedicated `interpret_objective` implementation;
do not guess roles from attribute names.

Installation and local execution do not by themselves establish behavioral or
scientific conclusions. Name the exact tests or experiment controls supporting
such claims.
