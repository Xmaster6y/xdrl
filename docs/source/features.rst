Features
========

TorchRL-native contracts
-------------------------

Declare actor, critic, value, loss, encoder, mixer, and world-model roles with
native TensorDict nested keys and TorchRL tensor specs. Batch semantics name
environment, time, agent, and objective axes without conflating them with
feature shape.

Typed execution contexts
------------------------

Describe collection, evaluation, replay, loss, target, and optimisation calls.
Contexts validate live inputs and outputs and restore exploration, gradient,
inference, autocast, and module train/evaluation state after success or failure.

TDHook observability and intervention
-------------------------------------

Bind TDHook factories to declared TensorDict interactions, with validated
module paths, explicit lazy materialisation, bounded typed observation traces,
spec-checked interventions, paired controls, and exception-safe cleanup.

Recurrent and multi-agent semantics
-----------------------------------

Represent recurrent state transitions, reset masks, sequence windows, agent
groups, parameter sharing, centralised critics, and mixers without treating a
shared PyTorch path as an agent identity. Unsupported collector lifecycles fail
explicitly.

Provenance and conformance
--------------------------

Record model/checkpoint identity, interaction semantics, selected keys and
paths, method configuration, dependency versions, and code revision. Separate
unit, integration, upstream-compatibility, and behavioural-parity suites define
the supported boundary.
