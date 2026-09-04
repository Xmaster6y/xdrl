Tutorials and reproductions
===========================

Choose a notebook based on what you want to learn.

Feature guides
--------------

Learn one XDRL capability at a time, in the suggested order.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Interpret one module
      :link: notebooks/collection
      :link-type: doc
      :class-card: surface

      Run an unchanged TorchRL policy through one interpreted component.

   .. grid-item-card:: Native TDHook workflow
      :link: notebooks/workflow-evidence
      :link-type: doc
      :class-card: surface

      Execute a TDHook workflow through an interpreted TorchRL component.

   .. grid-item-card:: Repeated module calls
      :link: notebooks/internal-computation
      :link-type: doc
      :class-card: surface

      Select a repeated call with TDHook's native occurrence support.

   .. grid-item-card:: Intervention
      :link: notebooks/intervention
      :link-type: doc
      :class-card: surface

      Apply a focused TDHook intervention through an interpreted component.

Complete workflows
------------------

Follow an end-to-end policy investigation that combines multiple capabilities.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: End-to-end investigation
      :link: notebooks/end-to-end-policy-investigation
      :link-type: doc
      :class-card: surface

      Run matched diagnosis and intervention workflows on one policy.

Paper-inspired examples
-----------------------

Learn the mechanics behind published interpretability methods on constructed examples.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Functional modules
      :link: reproductions/bixrl-functional-modularity
      :link-type: doc
      :class-card: surface

      Detect and prune modules in a synthetic classifier.

   .. grid-item-card:: Recurrent planning probes
      :link: reproductions/emergent-planning-sokoban
      :link-type: doc
      :class-card: surface

      Probe constructed future labels across recurrent calls.

   .. grid-item-card:: Multi-agent concept policies
      :link: reproductions/marl-concept-policy
      :link-type: doc
      :class-card: surface

      Intervene on concepts in a supervised policy example.

   .. grid-item-card:: Spatial goal steering
      :link: reproductions/maze-policy-goal-representations
      :link-type: doc
      :class-card: surface

      Patch engineered goal channels in an open-grid policy.

   .. grid-item-card:: Additive value decomposition
      :link: reproductions/na2q-value-decomposition
      :link-type: doc
      :class-card: surface

      Inspect unary and pairwise terms on generated tensors.

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
