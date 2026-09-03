---
name: xdrl-library
description: Build, review, or explain the typed boundary between TorchRL TensorDict modules and native TDHook workflows.
---

# XDRL library

Use XDRL for one concern: validating and scoping a TorchRL module call before
passing the unchanged module to TDHook.

## Ownership

- TensorDict and TorchRL own data, modules, specs, collectors, environments,
  replay, losses, and optimisation.
- TDHook owns model-internal targets, hooks, captures, replacements, workflows,
  planning, occurrences, artifacts, and cleanup.
- XDRL owns semantic input/output schemas, named batch dimensions, minimal
  recurrent boundary checks, and temporary Torch execution modes.
- Applications own experiment pairing, artifact metadata, reproducibility
  manifests, and scientific interpretation.

Do not introduce an XDRL hook implementation, workflow runner class,
provenance format, trainer, data container, or paired-experiment subsystem.

## Entry points

1. Declare one `InteractionSpec`.
2. Wrap the existing module once with `Interaction`.
3. Call the interaction directly for a normal TorchRL invocation.
4. Call `run_workflow(interaction, workflow, data)` for a TDHook workflow.
5. Use TDHook `Target(occurrence=...)` or `HookSession` directly for repeated
   model-internal calls.

`run_workflow` returns TDHook's native `WorkflowResult`.

<!-- runnable-example -->
```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow
from xdrl import BatchSemantics, Interaction, InteractionSpec, KeyRole
from xdrl import KeySchema, ModelRole, TensorDictSchema, run_workflow

policy = TensorDictModule(torch.nn.Linear(4, 2), ["observation"], ["action"])
spec = InteractionSpec(
    ModelRole.ACTOR,
    TensorDictSchema((KeySchema("observation", KeyRole.OBSERVATION),)),
    TensorDictSchema((KeySchema("action", KeyRole.ACTION),)),
    BatchSemantics(("env",)),
)
interaction = Interaction(policy, spec)
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
