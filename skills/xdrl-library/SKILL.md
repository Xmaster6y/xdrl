---
name: xdrl-library
description: Build, review, or explain typed XDRL integrations for TorchRL models with TensorDict schemas, TDHook observation or intervention, recurrent and multi-agent semantics, provenance, and compatibility evidence. Use when work involves the xdrl Python package, XDRL interaction contexts, TDHook adapters or planned workflows, or deciding whether an RL observability concern belongs in XDRL, TorchRL, TensorDict, or TDHook.
---

# XDRL library

Use XDRL as the typed boundary around an existing TorchRL model call. Keep data and execution in TensorDict/TorchRL, keep generic model-internal methods in TDHook, and use XDRL for interaction semantics, validation, lifecycle restoration, and provenance.

## Select the matching API reference

Check `xdrl.__version__` before writing code. For `0.2.x`, read [references/xdrl-0.2.md](references/xdrl-0.2.md) completely. For `0.1.x`, use the historical [references/xdrl-0.1.md](references/xdrl-0.1.md). For another version, stop and consult that version's released documentation; do not guess across versions.

Run `xdrl.validate_runtime_compatibility()` before describing a runtime as supported. A successful install or import proves resolution only, not API compatibility, behavioural parity, or conformance.

## Preserve ownership

- Leave environments, collectors, replay, loss modules, optimisation, tensor specs, nested keys, and exploration behavior in TorchRL/TensorDict.
- Leave activation capture, attribution, probing, patching, steering, hook registration, workflow planning, artifacts, and pass counts in TDHook.
- Use XDRL for model roles, schemas, interaction phases, execution-state restoration, observation/intervention records, recurrent and multi-agent meaning, compatibility boundaries, and provenance.

Do not introduce an XDRL trainer, data container, tensor-spec hierarchy, or duplicate TDHook method.

## Build an interaction

1. Declare input and output `TensorDictSchema` objects with semantic key roles, presence, native nested keys, TorchRL specs when needed, and named batch dimensions.
2. Snapshot those schemas into an `InteractionDescriptor`. Record a stable interaction identity, model role, phase, module path, model/checkpoint identity, and exact execution modes.
3. Wrap the existing TensorDict-compatible module in `RuntimeInteractionContext`. Use the one-shot call for local synchronous execution, or keep the context open when hooks must survive through backward.
4. Attach an `ObservationTrace` for bounded metadata or opt-in tensor retention. Observation alone must preserve output behavior.
5. Make every intervention explicit about target, timing, scope, and replacement/transform. Use `run_paired` for matched baseline/intervention mechanics; do not turn mechanical differences into causal claims.
6. Compose TDHook methods in a public `Workflow`, then use `TDHookWorkflowRunner.plan` and `run`. Use TDHook's `Target` and `HookSession` for interactive model-internal interventions. Never regroup executions or reinterpret TDHook's planned pass count.
7. Retain `ProvenanceManifest` records with the model/checkpoint, descriptor, selected keys, resolved target paths, method configuration, dependency versions, and code revision.

## Treat advanced semantics explicitly

- Declare recurrent state as required input and produced next-state keys. Name reset masks and sequence dimensions. Only direct, synchronous, and replay-sequence lifecycles are validated.
- Target multi-agent work by semantic group, agent selection, model role, and topology. Never use a shared module path as an agent identity.
- Fail closed for compiled, remote, distributed, multiprocessing, async-collector, and worker-copied execution unless the selected version's conformance contract explicitly supports it.

## Report evidence precisely

Separate these statements:

- package installation/import succeeded;
- `validate_runtime_compatibility()` accepted the dependency boundary;
- a named unit, integration, upstream-compatibility, or behavioural-parity suite passed;
- the requested execution mode is supported, experimental, or unsupported;
- an observed behavioral or scientific conclusion is justified by experiment evidence.

Name the exact conformance suite behind any support claim. Preserve failures and unsupported-mode errors rather than weakening the declared boundary.
