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

.. dependency-snapshot: start

==========  ======================  ====================================================================================================================================================================================================================================
Component   Tested requirement      Evidence
==========  ======================  ====================================================================================================================================================================================================================================
Python      ``>=3.11,<3.14``        required CI matrix
PyTorch     ``==2.13.*``            compatibility and parity suites; ``((python_full_version >= '3.12' and sys_platform != 'darwin') or (python_full_version == '3.12.*' and sys_platform == 'darwin')) or (python_full_version < '3.12' and sys_platform != 'darwin')``
PyTorch     ``==2.14.*``            compatibility and parity suites; ``(python_full_version >= '3.13' and sys_platform == 'darwin') or (python_full_version < '3.12' and sys_platform == 'darwin')``
TensorDict  ``==0.13.0+g54a147b``   schema and parity suites
TorchRL     ``==0.13.0+gae421b98``  compatibility and integration
TDHook      ``==0.2.0``             workflow conformance suite
xdrl        ``==0.2.0``             all required suites
==========  ======================  ====================================================================================================================================================================================================================================

.. dependency-snapshot: end

The current TorchRL and TDHook sources are Git revisions recorded in
``uv.lock``. A newly installable upstream revision remains experimental until
the matrix is updated and all required gates pass.

Development dependency policy
-----------------------------

XDRL currently advances with TDHook and TorchRL development. ``tool.uv.sources``
therefore follows their ``main`` branches while ``uv.lock`` records the exact
commits exercised by CI. Updating either source requires refreshing the lock,
the compatibility declarations below, and every required conformance and
documentation gate in the same change. Branch tracking is a development policy,
not a claim that arbitrary upstream commits are supported.

A future public release must replace this source-tracked policy with dependency
metadata that reproduces the tested runtime from an index installation. That
transition is deferred until the APIs and dependency versions are ready to
stabilise.

Workflow conformance
--------------------

``TDHookWorkflowRunner`` is supported by
``tests/behavioural_parity/test_tdhook_workflow.py``. Its conformance contract
covers TensorDict schema preservation, deterministic observation-only output
parity, lifecycle cleanup, exception safety, explicit lazy materialisation,
plan delegation, and model-pass accounting.
``tests/integration/test_tdhook_workflow.py`` owns the declared-workflow
boundary: TDHook-controlled coexecution, schema validation on every actual
model call, module identity preservation, plan agreement, and cleanup.
Compiled, distributed, remote, multiprocess, async, and distributed-collector
paths remain unsupported and are rejected where they reach a known boundary.

Test ownership is separated deliberately:

* ``tests/unit`` checks local data and validation contracts.
* ``tests/upstream_compatibility`` owns version-sensitive and private APIs.
* ``tests/integration`` checks cross-package composition and provenance.
* ``tests/behavioural_parity`` compares native and instrumented execution.

Private upstream APIs
---------------------

``PRIVATE_UPSTREAM_APIS`` is the machine-readable inventory. The only owned
use is Torch's ``_orig_mod`` compiled-module marker, which lets the TDHook
adapter reject compiled descendants before installing hooks. Its lockstep test
is ``tests/upstream_compatibility/test_private_apis.py``; adding another private
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
