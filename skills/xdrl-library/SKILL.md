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

Installation and local execution do not by themselves establish behavioral or
scientific conclusions. Name the exact tests or experiment controls supporting
such claims.
