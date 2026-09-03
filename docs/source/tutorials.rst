Tutorials and reproductions
===========================

Choose a path based on what you want to learn. If you are new to XDRL, begin
with the feature guides and follow them in order. Complete workflows combine
several features into a realistic investigation, while scientific
reproductions evaluate claims from published work with their own provenance,
controls, evidence status, and claim limits.

Feature guides
--------------

These focused notebooks introduce one XDRL boundary or capability at a time,
progressing from an unchanged policy interaction to model-internal observation
and intervention.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: One interaction
      :link: notebooks/collection
      :link-type: doc
      :class-card: surface

      Run an unchanged TorchRL policy through one interaction.

   .. grid-item-card:: Native TDHook workflow
      :link: notebooks/workflow-evidence
      :link-type: doc
      :class-card: surface

      Execute a TDHook workflow through XDRL's single workflow entrypoint.

   .. grid-item-card:: Repeated module calls
      :link: notebooks/internal-computation
      :link-type: doc
      :class-card: surface

      Select a repeated call with TDHook's native occurrence support.

   .. grid-item-card:: Intervention
      :link: notebooks/intervention
      :link-type: doc
      :class-card: surface

      Apply a focused TDHook intervention through an XDRL interaction.

Complete workflows
------------------

These task-oriented notebooks combine multiple capabilities into an end-to-end
policy investigation.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: End-to-end investigation
      :link: notebooks/end-to-end-policy-investigation
      :link-type: doc
      :class-card: surface

      Run matched diagnosis and intervention workflows on one policy.

Scientific reproductions
------------------------

These paper-specific notebooks keep scientific evaluation distinct from
product tutorials. Each reproduction states its provenance, controls, evidence
status, and the limits of the claims it can support.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: BiXRL functional modularity
      :link: reproductions/bixrl-functional-modularity
      :link-type: doc
      :class-card: surface

      Exercise functional-modularity analysis on a bounded fixture.

   .. grid-item-card:: Emergent planning in Sokoban
      :link: reproductions/emergent-planning-sokoban
      :link-type: doc
      :class-card: surface

      Probe repeated recurrent computations using occurrence selection.

   .. grid-item-card:: MARL concept policies
      :link: reproductions/marl-concept-policy
      :link-type: doc
      :class-card: surface

      Evaluate concept interventions with explicit agent axes.

   .. grid-item-card:: Maze goal representations
      :link: reproductions/maze-policy-goal-representations
      :link-type: doc
      :class-card: surface

      Recover and steer a spatial representation with matched controls.

   .. grid-item-card:: NA2Q value decomposition
      :link: reproductions/na2q-value-decomposition
      :link-type: doc
      :class-card: surface

      Preserve agent and coalition structure through a mixer workflow.

.. toctree::
   :hidden:

   notebooks/collection.ipynb
   notebooks/workflow-evidence.ipynb
   notebooks/internal-computation.ipynb
   notebooks/intervention.ipynb
   notebooks/end-to-end-policy-investigation.ipynb
   reproductions/bixrl-functional-modularity.ipynb
   reproductions/emergent-planning-sokoban.ipynb
   reproductions/marl-concept-policy.ipynb
   reproductions/maze-policy-goal-representations.ipynb
   reproductions/na2q-value-decomposition.ipynb
