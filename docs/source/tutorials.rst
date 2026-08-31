Tutorials
=========

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Local collection
      :link: notebooks/collection
      :link-type: doc
      :class-card: surface

      :octicon:`sync;2em;sd-text-primary`

      Run a typed policy in TorchRL's local collector.

   .. grid-item-card:: Intervention
      :link: notebooks/intervention
      :link-type: doc
      :class-card: surface

      :octicon:`tools;2em;sd-text-primary`

      Compare baseline and intervened policy actions.

   .. grid-item-card:: Workflow evidence
      :link: notebooks/workflow-evidence
      :link-type: doc
      :class-card: surface

      :octicon:`pulse;2em;sd-text-primary`

      Cache an activation and inspect execution provenance.

   .. grid-item-card:: Repeated internal computation
      :link: notebooks/internal-computation
      :link-type: doc
      :class-card: surface

      :octicon:`iterations;2em;sd-text-primary`

      Record semantic coordinates for calls to a reused module.

.. toctree::
   :hidden:

   notebooks/collection.ipynb
   notebooks/internal-computation.ipynb
   notebooks/intervention.ipynb
   notebooks/workflow-evidence.ipynb

Reproductions
-------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Functional neural modules in MiniGrid
      :link: reproductions/bixrl-functional-modularity
      :link-type: doc
      :class-card: surface

      :octicon:`beaker;2em;sd-text-primary`

      BIXRL 2D module detection and matched weight interventions.

      **Evidence status:** bounded smoke path only; no scientific agreement claim.

.. toctree::
   :hidden:

   reproductions/bixrl-functional-modularity.ipynb
