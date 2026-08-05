Features
========

XDRL adds a small typed boundary around TorchRL model calls:

* **Contracts:** native TensorDict keys, TorchRL specs, model roles, phases,
  and named batch dimensions.
* **Runtime contexts:** input/output validation with restoration of module,
  gradient, inference, autocast, and exploration state.
* **TDHook workflows:** validated targets, observation and intervention,
  planned model-pass counts, and cleanup after failures.
* **RL semantics:** optional recurrent state, reset/window, and multi-agent
  topology declarations.
* **Provenance:** the interaction, workflow plan, dependency versions, code
  revision, and observed model calls.
