Retiring the ``news`` design spike
==================================

This is the final disposition report for GitHub issue #24.  The retired
``news`` branch was exploratory design evidence, not an implementation source.
This report audits ``origin/main...origin/news`` at ``0f4d9f5`` and records
what may be reused without merging or cherry-picking the branch.

Decision rule
-------------

The project boundary in the :doc:`architecture` governs every disposition:
TorchRL owns trainers, collectors, environments, logging, checkpointing, and
configuration-driven experiments; TDHook owns generic model-internal methods;
``xdrl`` owns typed interaction contracts and their TDHook binding.  A useful
experiment is retained only as a requirement, independent conformance test,
or upstream report within that boundary.  No API, configuration, or behavioural
compatibility with the retired spike is promised.

Branch-only disposition
-----------------------

* ``configs/gymnasium_dqn.yaml`` is **category 5 / discard**: its TrackIO
  logger selection and hook wiring are experiment configuration, not a core
  contract.
* ``configs/mogymnasium_ppo.yaml`` is **category 5 / discard**: MO-Gymnasium
  batching and logger choices are experiment configuration and environment
  setup.
* ``configs/vmas_ppo.yaml`` and ``configs/vmas_qmix.yaml`` are **category 5 /
  discard**: their TrackIO migration and trainer-hook wiring are outside
  xdrl's core scope.
* ``configs/xdrl_hook_stack/*`` is **category 5 / discard**: evaluation, W&B,
  TrackIO, and checkpoint presets would make xdrl a trainer framework.
* ``pyproject.toml`` and ``uv.lock`` are **category 4 / discard**: the switch
  to a moving upstream TorchRL source is not a supported workaround.  The
  pinned, test-backed revision in :doc:`compatibility` remains the only
  supported boundary.
* ``scripts/run_experiment.py`` is **category 5 / discard**: warning
  suppression is a local experiment concern and must not define library
  behaviour.
* ``src/xdrl/configs/__init__.py`` is **category 5 / discard**: TrackIO
  registration belongs to a trainer integration, not the interaction API.
* ``src/xdrl/configs/hooks.py`` is **category 5 / discard**: progress metrics,
  validation lifecycle changes, and checkpoint scheduling extend trainer
  policy.
* ``src/xdrl/envs.py`` is **category 4 / discard**: the MO-Gymnasium warning
  filter/wrapper is an environment workaround and has no core owner.
* ``src/xdrl/trainer_hooks/*`` is **category 5 / discard**: logger cleanup,
  progress metrics, and target-frame checkpoints are trainer framework work.
* ``tests/unit/test_configs.py`` and ``tests/unit/test_trainer_hooks.py`` are
  **category 5 / discard**: they validate the discarded experiment
  configuration, logging integration, trainer lifecycle, and checkpoint
  features.

The branch also motivated the requirement that model observation and
intervention be described at TorchRL/TensorDict interaction boundaries instead
of through a replacement trainer stack.  That requirement is already captured
by roadmap issues #17--#23, and the planned-workflow boundary is tracked by
#34.  No implementation from ``news`` is needed for those issues.

Retained upstream evidence
--------------------------

``tests/unit/test_torchrl_failures.py`` contains five focused, strict-xfail
reproductions.  They are valuable as upstream compatibility evidence, but do
not establish local support or justify local patches.  Follow-up issue #35
owns reducing them against the pinned revision and filing (or explicitly
retiring) reports for:

* continuous ``SACLossConfig`` forwarding discrete-only fields;
* an obsolete ``TransformedEnv`` auto-unwrap warning;
* vector Gymnasium autoreset reward shape mixing;
* ``LogValidationReward`` closing a reusable validation environment; and
* Gymnasium pixel detection importing legacy Gym/CV2.

They are intentionally not copied into xdrl's supported conformance suite.
That suite protects the declared, version-pinned adapter and private-API
boundary described in :doc:`compatibility`; upstream defects require their own
version-qualified reports.

Local bridge experiment
-----------------------

At audit time the working tree has no tracked or untracked source changes
(``git status --short`` is empty).  The two pre-existing stashes are historical
``bugs``-branch WIP entries, not changes on the retirement branch, and are
retained untouched because deletion is outside this issue.  Ignored build,
cache, virtual-environment, and experiment-output paths are not source work
and are likewise left untouched.  There is therefore no uncommitted bridge
implementation to merge, preserve, or discard from the active worktree.

Completion and deletion boundary
--------------------------------

This report, #35, and the already-linked roadmap issues are the preserved audit
result.  The remote and local ``news`` refs were deleted after this audit was
reviewed.  No branch history, implementation, or compatibility commitment is
retained beyond the explicitly scoped requirements and upstream reports above.
