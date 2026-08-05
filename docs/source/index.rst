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
    architecture
    compatibility
    migration
    news-retirement
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

        **xdrl** provides typed interaction contracts and TDHook bindings for
        TorchRL models. TorchRL continues to own trainer execution, logging,
        evaluation, checkpointing, and environments; xdrl makes model-internal
        observation and intervention explicit at those interaction boundaries.

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

          **Typed interaction contracts**

          Describe model roles, TensorDict schemas, execution phases, and
          batch semantics without replacing TorchRL's trainer lifecycle.

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/two.png
          :width: 150

        .. div::

          **TDHook-powered interpretability**

          Observe and intervene inside declared TorchRL model interactions with
          TDHook while preserving native TensorDict behaviour.
