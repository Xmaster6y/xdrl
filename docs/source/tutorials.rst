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

   .. grid-item-card:: End-to-end policy investigation
      :link: notebooks/end-to-end-policy-investigation
      :link-type: doc
      :class-card: surface

      :octicon:`workflow;2em;sd-text-primary`

      Connect competence, attribution, matched interventions, and behavior.

.. toctree::
   :hidden:

   notebooks/collection.ipynb
   notebooks/end-to-end-policy-investigation.ipynb
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

   .. grid-item-card:: Emergent planning in Sokoban
      :link: reproductions/emergent-planning-sokoban
      :link-type: doc
      :class-card: surface

      :octicon:`iterations;2em;sd-text-primary`

      Exact DRC occurrences, spatial probes, and matched intervention controls.

      **Evidence status:** bounded smoke path only; paper-exact assets unavailable.

   .. grid-item-card:: Goal representations in Procgen Maze
      :link: reproductions/maze-policy-goal-representations
      :link-type: doc
      :class-card: surface

      :octicon:`goal;2em;sd-text-primary`

      Reported-channel localization, declared controls, and matched steering effects.

      **Evidence status:** bounded smoke path only; checkpoint provenance is incomplete.

   .. grid-item-card:: NA2Q value decomposition in cooperative MARL
      :link: reproductions/na2q-value-decomposition
      :link-type: doc
      :class-card: surface

      :octicon:`people;2em;sd-text-primary`

      Named unary/pair coalitions, identity masks, and additive joint-value parity.

      **Evidence status:** bounded smoke path only; paper checkpoints and result data are unavailable.

   .. grid-item-card:: Expert-concept policies in simulated MARL
      :link: reproductions/marl-concept-policy
      :link-type: doc
      :class-card: surface

      :octicon:`comment-discussion;2em;sd-text-primary`

      Matched concept-policy controls and frozen concept-correction interventions.

      **Evidence status:** bounded synthetic smoke path only; FortAttack assets are unavailable.

.. toctree::
   :hidden:

   reproductions/bixrl-functional-modularity.ipynb
   reproductions/emergent-planning-sokoban.ipynb
   reproductions/maze-policy-goal-representations.ipynb
   reproductions/marl-concept-policy.ipynb
   reproductions/na2q-value-decomposition.ipynb
