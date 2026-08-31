Repeated internal computation
=============================

Some recurrent policies perform several model-internal updates for one
observed environment state.  XDRL represents those updates without treating a
notebook loop index or forward-hook counter as scientific identity.

The contract separates five concerns:

* ``InteractionContract.time_dimension`` names environment time.
* ``RecurrentSemantics`` owns sequence, burn-in, truncation, reset, and state
  transition semantics.
* ``InternalComputationAxis`` names architecture-facing dimensions such as
  ``tick`` and ``layer``.
* ``InternalOccurrence`` maps one tuple of semantic coordinates to a raw,
  zero-based hook-call index for one module path within each root call.
* ``logical_step`` and lifecycle event order describe root interaction calls;
  neither is an internal tick.

The mapping is deliberately explicit.  Reusing one module instance for several
layers or ticks therefore produces distinct occurrences instead of silently
overwriting one activation.

Declaring and selecting occurrences
-----------------------------------

.. code-block:: python

   from xdrl import (
       InternalComputationAxis,
       InternalComputationSemantics,
       InternalOccurrence,
       InternalOccurrenceSelection,
   )

   semantics = InternalComputationSemantics(
       axes=(
           InternalComputationAxis("tick", (0, 1)),
           InternalComputationAxis("layer", ("lower", "upper")),
       ),
       occurrences=(
           InternalOccurrence("module.cell", 0, (0, "lower")),
           InternalOccurrence("module.cell", 1, (0, "upper")),
           InternalOccurrence("module.cell", 2, (1, "lower")),
           InternalOccurrence("module.cell", 3, (1, "upper")),
       ),
       recurrent_state_keys=(("state",), ("next", "state")),
   )
   second_tick = semantics.select(
       InternalOccurrenceSelection((("tick", 1),))
   )

Attach ``semantics`` as ``InteractionContract.internal_computation``.  The
contract requires recurrent semantics, checks that the named state keys belong
to its declared transitions, and forbids an internal axis from reusing the
environment/sequence time name.

Runtime evidence and failure behaviour
--------------------------------------

``RuntimeInteractionContext.observe_internal_computation()`` installs
temporary public PyTorch forward hooks around the declared targets.  For every
root interaction call it resets raw counters, records the exact semantic
coordinates on ``ObservationRecord.internal_coordinates``, and verifies the
complete call mapping before the root result is accepted.  Undeclared, missing,
extra, or nested calls raise ``OccurrenceIdentityError``.  Hooks are removed on
success and failure.  Module and callable overrides are rejected while the
observer is active because its hooks are bound to the contract's declared root
module.

This observer supports local synchronous activation evidence.  It does not
claim that hook order is a portable model identity and it does not implement an
internal intervention engine.

TDHook dependency
-----------------

Activation and gradient interventions remain owned by TDHook.  Associating
them with these coordinates requires TDHook to expose a public occurrence
selector and execution evidence that reports the selected module path and raw
call index.  The currently supported TDHook revision does not provide that
capability.  ``TDHookWorkflowRunner`` therefore rejects contracts containing
``internal_computation`` before planning or execution, rather than accepting an
ambiguous intervention or producing scientific artifacts with collapsed
identity.  Once TDHook supplies that public contract, XDRL can resolve an
``InternalOccurrenceSelection`` to its exact raw occurrences and validate the
returned evidence.  The generic upstream work is tracked in `TDHook issue 128
<https://github.com/Xmaster6y/tdhook/issues/128>`_.

Provenance
----------

Workflow provenance schema revision 4 serializes axes, coordinates, raw
occurrence mappings, and recurrent-state linkage as tensor-free contract data.
Decoding is strict: missing fields, unknown fields, invalid coordinates, or a
mapping that violates the canonical interaction invariants are rejected.
