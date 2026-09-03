About ``xdrl``
==============

``xdrl`` provides interpretability extensions for TorchRL modules and
objectives.

TorchRL and TensorDict own RL execution, data, algorithms, and parameters.
TDHook owns generic PyTorch hooks and model-internal methods. XDRL discovers
the RL meaning already expressed by TorchRL objects and executes those methods
against the correct actor, critic, value, Q-value, mixer, online, or target
parameterization.

Project links
-------------

* `Source code <https://github.com/Xmaster6y/xdrl>`_
* `PyPI package <https://pypi.org/project/xdrl/>`_
* `MIT License <https://github.com/Xmaster6y/xdrl/blob/main/LICENSE>`_

Literature
----------

The references below cover the foundations cited by XDRL's public API and the
papers targeted by its :doc:`reproduction notebooks <tutorials>`. Each
reproduction notebook separately records the exact paper, reference-code
revision, asset availability, and limits of the evidence it produces.

References
----------

.. bibliography::
   :all:
