<div align="center">
<img src="https://raw.githubusercontent.com/Xmaster6y/xdrl/refs/heads/main/docs/source/_static/images/xdrl-logo.png" alt="logo" width="200"/>
</div>

<h1 align=center><code>xdrl</code> 🔍</h1>

[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://xdrl.readthedocs.io)
[![xdrl](https://img.shields.io/pypi/v/xdrl?color=purple)](https://pypi.org/project/xdrl/)
[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/Xmaster6y/xdrl/blob/main/LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![python versions](https://img.shields.io/pypi/pyversions/xdrl.svg)](https://www.python.org/downloads/)

[![codecov](https://codecov.io/gh/Xmaster6y/xdrl/graph/badge.svg)](https://codecov.io/gh/Xmaster6y/xdrl)
![ci](https://github.com/Xmaster6y/xdrl/actions/workflows/ci.yml/badge.svg)
[![docs](https://readthedocs.org/projects/xdrl/badge/?version=latest)](https://xdrl.readthedocs.io/en/latest/?badge=latest)

Interpretability extensions for [TorchRL](https://github.com/pytorch/rl).

XDRL discovers the actor, critic, value functions, Q-value ensembles, mixers,
and online or target parameterizations already present in native TorchRL
objects. It then exposes those components to
[TDHook](https://github.com/Xmaster6y/tdhook) without asking you to describe the
RL system a second time.

## Getting Started

```bash
pip install xdrl
```

```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow
from xdrl import interpret

policy = TensorDictModule(
    torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 2),
    ),
    in_keys=["observation"],
    out_keys=["action"],
)
batch = TensorDict(
    {"observation": torch.randn(8, 4)},
    batch_size=[8],
)
policy = interpret(policy)
workflow = Workflow(
    ActivationCaching("module.1", cache_key=("activations", "hidden"))
)
execution = policy.run(workflow, batch)

assert execution.data["action"].shape == (8, 2)
assert execution.data["activations", "hidden", "module.1"].shape == (8, 8)
```

TorchRL and TensorDict own policy execution, keys, specs, parameters, and
batched data. TDHook owns the model-internal method: here, capturing the hidden
activation. XDRL's `interpret` view connects them and validates the call
boundary. An activation capture records an internal value; by itself, it is not
evidence that the activation causally affects the policy's action.

XDRL also understands native TorchRL objectives. For example,
`interpret(SACLoss(...))` exposes `.actor`, each member of `.qvalue`, and each
member of `.target.qvalue` with the correct functional parameters already
bound. Native probabilistic actors, Q-value actors, value operators, and
actor-value operators expose their existing RL functions in the same way;
plain `TensorDictModule` objects remain directly executable components.
Explicit objective integrations are included for DQN, PPO, SAC, IQL, and
QMixer.

For recurrent TorchRL modules, see `RecurrentSemantics` in the
[API reference](https://xdrl.readthedocs.io/en/latest/api/index.html).

## Development

This project uses [`uv`](https://docs.astral.sh/uv/) to manage Python
dependencies and [`just`](https://github.com/casey/just) to run the
test and documentation gates.

## Documentation

- [Getting Started](https://xdrl.readthedocs.io/en/latest/start.html)
- [Tutorials](https://xdrl.readthedocs.io/en/latest/tutorials.html)
- [API Reference](https://xdrl.readthedocs.io/en/latest/api/index.html)
- [About](https://xdrl.readthedocs.io/en/latest/about.html)

## License
`xdrl` is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
