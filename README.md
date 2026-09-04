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

Use [TDHook](https://github.com/Xmaster6y/tdhook) methods on native
[TorchRL](https://github.com/pytorch/rl) modules and losses.

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

model = TensorDictModule(
    torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 2),
    ),
    in_keys=["observation"],
    out_keys=["action"],
)
policy = interpret(model)
batch = TensorDict({"observation": torch.randn(8, 4)}, batch_size=[8])
result = policy.run(
    Workflow(ActivationCaching("module.1", cache_key=("activations", "hidden"))),
    batch,
)

assert result.data["action"].shape == (8, 2)
assert result.data["activations", "hidden", "module.1"].shape == (8, 8)
```

`interpret` preserves the TensorDict API and adds `.run(...)` for TDHook
workflows. The policy, TensorDict keys, and execution behavior stay native to
TorchRL; XDRL only supplies the interpretability view.

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
