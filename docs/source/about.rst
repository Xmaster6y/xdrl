About `xdrl`
============

``xdrl`` is an alpha-stage research library for making model-internal work on
TorchRL systems explicit and reproducible. It supplies the RL-specific type,
schema, execution-context, and provenance layer needed to apply TDHook methods
to TensorDict-native models.

The project intentionally does not provide another RL trainer or copy generic
interpretability algorithms. TorchRL and TensorDict remain responsible for RL
execution and data; TDHook remains responsible for PyTorch hooks and methods;
``xdrl`` owns the validated boundary between them.

Support claims are evidence-based. A dependency being importable, a hook
running once, or an example configuration existing does not establish library
support. See :doc:`compatibility` for the tested matrix and conformance suites.
