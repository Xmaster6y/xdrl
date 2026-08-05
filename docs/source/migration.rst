Migrating to XDRL 0.2
=====================

XDRL 0.2 is an intentional breaking release aligned with TDHook 0.2. It removes
the compatibility layer for TDHook's retired ``Pipeline`` API and makes
ownership boundaries explicit.

Interaction contracts
---------------------

Replace ``InteractionDescriptor`` and its separate schema snapshots with one
``InteractionContract`` containing the live input and output
``TensorDictSchema`` values. Batch semantics are derived from those schemas,
which must agree. Construct ``RuntimeInteractionContext`` from the contract,
module, and representative input; the context no longer accepts duplicate
schema arguments::

   contract = InteractionContract(
       "policy:evaluation:0",
       ModelRole.ACTOR,
       InteractionPhase.EVALUATION,
       "policy",
       input_schema,
       output_schema,
   )
   interaction = RuntimeInteractionContext(contract, policy, representative_input)

Workflow execution
------------------

Replace ``TDHookInteractionAdapter.run_pipeline`` with a TDHook ``Workflow``
and ``TDHookWorkflowRunner``::

   workflow = Workflow(ActivationCaching("module"))
   runner = TDHookWorkflowRunner(interaction)
   plan = runner.plan(workflow, batch)
   execution = runner.run(
       workflow, batch, code_revision="your-git-revision", expected_plan=plan
   )

TDHook owns planning, method compatibility, coexecution, artifacts, hook
programs, and cleanup. XDRL validates the typed RL boundary around every actual
root model call and checks the observed call count against
``WorkflowPlan.model_passes``. XDRL no longer mutates a module's Python class to
intercept calls.

``WorkflowRunRecord`` and ``ProvenanceManifest`` have been replaced by the
single versioned ``WorkflowProvenance`` returned as ``execution.provenance``.
It records a tensor-free contract projection, the public TDHook plan evidence,
lifecycle events, dependency versions, the required code revision, and an
optional seed. Invalid reproduction metadata is rejected before execution.

Model-internal interventions
----------------------------

``TDHookInterventionFactory`` and XDRL activation or gradient targets have been
removed. Use TDHook ``Target`` and ``HookSession`` for activation, gradient, and
parameter work. XDRL's ``Intervention`` remains for declared TensorDict input
and output keys, where recurrent and multi-agent semantics are meaningful.

Dependency boundary
-------------------

XDRL 0.2 requires ``tdhook>=0.2,<0.3``. Run
``validate_runtime_compatibility()`` and the relevant conformance suite before
describing an environment as supported; successful dependency resolution alone
is not behavioural evidence.
