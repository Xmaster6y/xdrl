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
    API Reference <api/index>
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

        Typed, inspectable TorchRL model interactions with TDHook workflows.

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

  What XDRL adds

.. grid:: 1 1 2 2
    :class-container: features

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/one.png
          :width: 150

        .. div::

          **Typed interaction contracts**

          Declare model roles, TensorDict schemas, phases, and batch semantics.

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/two.png
          :width: 150

        .. div::

          **TDHook-powered interpretability**

          Run TDHook workflows against declared TorchRL model interactions.
