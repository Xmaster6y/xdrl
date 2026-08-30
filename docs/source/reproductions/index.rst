Reproductions
=============

Paper reproductions live here as notebooks, separately from :doc:`../tutorials`.

.. important::

   No reproduction has been selected yet. See `issue #48
   <https://github.com/Xmaster6y/xdrl/issues/48>`_.

Evidence status
---------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Field
     - Status
   * - Smoke execution
     - ``not run``, ``passed``, or ``failed``
   * - Experiment artifacts
     - ``incomplete`` or ``complete``
   * - Reference agreement
     - ``not assessed``, ``agrees``, ``disagrees``, or ``inconclusive``
   * - Broader paper claims
     - ``out of scope``, ``partially assessed``, or ``assessed``

These fields are independent. Rendering, smoke execution, and completed
artifacts do not imply reference agreement or validate broader paper claims.

Gallery
-------

No entries yet. Each future card links one primary notebook, the paper,
reference code, required assets, artifacts, evidence status, and claim limits.

Primary notebook header
-----------------------

Start each notebook with one Markdown cell:

.. code-block:: markdown

   # <Paper reproduction title>

   - Paper: <citation and link>
   - Reference code: <repository link and immutable revision>
   - Source revisions: XDRL `<sha>`; TDHook `<sha>`; <other code/assets>
   - Execution: `<smoke|full>`; <hardware and configuration>
   - Assets: <source, revision, checksum, and local path>
   - Evidence: <the four statuses above and artifact links>
   - Claim limits: <what this notebook does and does not establish>

Link the notebook from its gallery card so ``just docs`` checks it. The docs
build renders stored cells without executing them.
