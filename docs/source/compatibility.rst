Compatibility, provenance, and conformance
===========================================

Installing xdrl and its dependencies establishes only that the packages can be
resolved. It does not establish API compatibility or behavioural support.
``xdrl.compatibility`` exposes the test-backed boundary used by this project,
and ``validate_runtime_compatibility()`` reports a named version or dependency
boundary when the active runtime falls outside it.

Support definitions
-------------------

.. list-table:: Evidence levels
   :header-rows: 1

   * - Level
     - Meaning
   * - Supported
     - The full named conformance suite passes in every required CI version.
   * - Experimental
     - Unit and integration evidence exists, but the full supported matrix is
       not a required gate.
   * - Unsupported
     - The boundary is rejected explicitly or has no advertised conformance
       suite.

Tested version matrix
---------------------

The lockfile is the reproducible source of exact revisions. The public runtime
contract accepts only the following tested versions:

================  =========================  ================================
Component         Tested requirement         Evidence
================  =========================  ================================
Python            ``>=3.11,<3.14``           required CI matrix
PyTorch           ``2.11.*``                 compatibility and parity suites
TensorDict        ``0.12.2``                 schema and parity suites
TorchRL           ``0.12.0+g5b2bc08b``       compatibility and integration
TDHook            ``0.1.3.dev0``             adapter conformance suite
xdrl              ``0.1.0``                  all required suites
================  =========================  ================================

The current TorchRL and TDHook sources are Git revisions recorded in
``uv.lock``. A newly installable upstream revision remains experimental until
the matrix is updated and all required gates pass.

Adapter conformance
-------------------

``TDHookInteractionAdapter`` is supported by
``tests/behavioural_parity/test_tdhook_adapter.py``. Its conformance contract
covers TensorDict schema preservation, deterministic observation-only output
parity, lifecycle cleanup, exception safety, explicit lazy materialisation, and
target-path resolution. Compiled, distributed, remote, multiprocess, async,
and distributed-collector paths remain unsupported and are rejected where they
reach a known boundary.

Test ownership is separated deliberately:

* ``tests/unit`` checks local data and validation contracts.
* ``tests/upstream_compatibility`` owns version-sensitive and private APIs.
* ``tests/integration`` checks cross-package composition and provenance.
* ``tests/behavioural_parity`` compares native and instrumented execution.

Private upstream APIs
---------------------

``PRIVATE_UPSTREAM_APIS`` is the machine-readable inventory. The owned uses are
Torch's ``_orig_mod`` compiled-module marker and TorchRL's ``Trainer._log``,
``_normalize_hydra_key``, and ``_resolve_module`` surfaces, plus TorchRL's W&B
logger ``_step_registry``. Their lockstep test is
``tests/upstream_compatibility/test_private_apis.py``; adding another private
surface requires adding it to both the inventory and that owner suite.

Provenance manifests
--------------------

``ProvenanceManifest`` records the model and checkpoint identifiers, complete
serialised interaction descriptor, selected TensorDict keys, resolved TDHook
paths and method configuration, exploration and gradient modes, batch
semantics, dependency versions, and code revision. ``to_json()`` is
deterministic and ``from_json()`` rejects unknown schema revisions, missing or
unknown fields, incomplete dependency provenance, and non-serialisable method
configuration. A manifest establishes reproducibility metadata; it does not by
itself establish that an adapter is supported.

Documentation examples are built in the same required CI workflow as the test
gates. Examples must name their support level and conformance suite; dependency
installation alone must never be described as support.
