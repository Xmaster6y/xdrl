Reproductions
=============

This gallery is the documentation home for paper-specific scientific
reproductions. Tutorials remain task-oriented learning material under
:doc:`../tutorials`; a tutorial notebook is never reused as a reproduction
deliverable.

.. important::

   No paper reproduction has been admitted yet. The candidate list is tracked
   in `issue #48 <https://github.com/Xmaster6y/xdrl/issues/48>`_. A paper appears
   here only after it has a focused reproduction issue and one dedicated primary
   notebook in this directory.

Evidence status
---------------

Every gallery entry reports four separate evidence fields. Later fields do not
follow automatically from earlier ones.

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Field
     - Allowed status
     - Meaning
   * - Smoke execution
     - ``not run``, ``passed``, or ``failed``
     - Whether the notebook completed a small maintenance run. A pass only
       checks that code paths execute with the stated smoke configuration.
   * - Experiment artifacts
     - ``incomplete`` or ``complete``
     - Whether the declared full experiment finished and its expected durable
       artifacts, manifest, and checksums are present.
   * - Reference agreement
     - ``not assessed``, ``agrees``, ``disagrees``, or ``inconclusive``
     - Whether a prespecified comparison against a cited reference result has
       been evaluated. Rendering, smoke execution, and artifact completion do
       not establish agreement.
   * - Broader paper claims
     - ``out of scope``, ``partially assessed``, or ``assessed``
     - Which claims, if any, the completed comparisons actually test. This field
       must link to the notebook's claim limits and may not generalize from one
       reproduced result to the full paper.

Gallery
-------

There are no reproduction entries yet. Each future entry must give the paper
and reference-code links, its single primary notebook link, required assets,
the four current evidence fields above, durable artifact links where available,
and a short claim-limits statement.

Primary notebook header
-----------------------

Start every primary notebook with one compact Markdown cell using this shape:

.. code-block:: markdown

   # <Paper reproduction title>

   - Paper: <citation and link>
   - Reference code: <repository link and immutable revision>
   - Source revisions: XDRL `<sha>`; TDHook `<sha>`; <other code/assets>
   - Execution mode: `<smoke|full>`; <hardware and configuration link>
   - Required assets: <source, version, checksum, and expected local path>
   - Evidence status:
     - Smoke execution: `<not run|passed|failed>`
     - Experiment artifacts: `<incomplete|complete>`; <manifest link>
     - Reference agreement: `<not assessed|agrees|disagrees|inconclusive>`
     - Broader paper claims: `<out of scope|partially assessed|assessed>`
   - Claim limits: <what this notebook does and does not establish>

Use immutable source revisions and link durable artifacts rather than relying on
the notebook's rendered outputs. Record deviations and failed or inconclusive
comparisons; do not promote them to success by changing the evidence labels.

Link every primary notebook from its gallery entry. The documentation build
follows those links and runs with warnings as errors. It renders stored cells
but does not execute them, so a green ``just docs`` build is documentation-link
evidence only.
