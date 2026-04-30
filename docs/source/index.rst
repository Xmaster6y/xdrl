:html_theme.sidebar_secondary.remove: true
:sd_hide_title:

`xdrl`
======

.. toctree::
    :maxdepth: 1
    :hidden:

    start
    features
    tutorials
    api/index
    About <about>

.. grid:: 1 1 2 2
    :class-container: hero
    :reverse:

    .. grid-item::
        .. div::

          .. image:: _static/images/xdrl-logo.png
            :width: 300
            :height: 300

    .. grid-item::

        .. div:: sd-fs-1 sd-font-weight-bold title-bot sd-text-primary image-container

            XDRL

        .. div:: sd-fs-4 sd-font-weight-bold sd-my-0 sub-bot image-container

            Hook-first interpretability for TorchRL

        **xdrl** provides a small hook foundation for TorchRL trainer loops.
        Today it covers logging, evaluation, GAE, validation, and policy checkpointing;
        planned work explores ``tdhook``-powered probing, steering, attribution, and representation analysis.

        .. div:: button-group

          .. button-ref:: start
            :color: primary
            :shadow:

                  Get Started

          .. button-ref:: tutorials
            :color: primary
            :outline:

                Tutorials

          .. button-ref:: api/index
            :color: primary
            :outline:

                API Reference


.. div:: sd-fs-1 sd-font-weight-bold sd-text-center sd-text-primary sd-mb-5

  Key Features

.. grid:: 1 1 2 2
    :class-container: features

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/one.png
          :width: 150

        .. div::

          **Trainer hooks**

          Attach metrics, validation, and checkpointing logic at TorchRL trainer lifecycle points.

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/two.png
          :width: 150

        .. div::

          **RL interpretability**

          Planned TensorDict-native probing, steering, attribution, and representation workflows for TorchRL trainers.
