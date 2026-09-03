---
name: xdrl-library
description: Build, review, or explain the integration between TorchRL TensorDict modules and native TDHook workflows.
---

# XDRL library

Use XDRL for one concern: validating RL relationships around a native TorchRL
module call before passing the unchanged module to TDHook.

## Ownership

- TensorDict and TorchRL own data, modules, specs, collectors, environments,
  replay, losses, and optimisation.
- TDHook owns model-internal targets, hooks, captures, replacements, workflows,
  planning, occurrences, artifacts, and cleanup.
- XDRL owns the interaction bridge and minimal recurrent boundary checks.
- Applications own experiment pairing, artifact metadata, reproducibility
  manifests, and scientific interpretation.

Do not introduce an XDRL hook implementation, workflow runner class,
provenance format, trainer, data container, or paired-experiment subsystem.

## Entry points

1. Wrap the existing module once with `Interaction`.
2. Call the interaction directly for a normal TorchRL invocation.
3. Call `run_workflow(interaction, workflow, data)` for a TDHook workflow.
4. Use TDHook `Target(occurrences=(...,))` or `HookSession` directly for repeated
   model-internal calls.

TensorDict's `batch_size` and dimension names describe the batch. The module's
`in_keys` and `out_keys` are the boundary. Use TorchRL `SafeModule` when values
need native spec enforcement. Use
`RecurrentSemantics.from_torchrl(...)` for TorchRL's `next` and `is_init`
conventions. The caller owns training, autograd, inference, exploration, and
autocast contexts; `run_workflow` checks autograd against TDHook's declared
requirements before execution.

`run_workflow` returns TDHook's native `WorkflowResult`.

<!-- runnable-example -->
```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow
from xdrl import Interaction, run_workflow

policy = TensorDictModule(torch.nn.Linear(4, 2), ["observation"], ["action"])
interaction = Interaction(policy)
data = TensorDict({"observation": torch.randn(8, 4)}, batch_size=[8])

native = interaction(data.clone())
result = run_workflow(
    interaction,
    Workflow(ActivationCaching("module", cache_key=("activations", "policy"))),
    data.clone(),
)

assert native["action"].shape == (8, 2)
assert result.plan.model_passes == 1
```

Installation and local execution do not by themselves establish behavioral or
scientific conclusions. Name the exact tests or experiment controls supporting
such claims.
