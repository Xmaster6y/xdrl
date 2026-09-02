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

Typed, inspectable [TorchRL](https://github.com/pytorch/rl) model interactions
with [TDHook](https://github.com/Xmaster6y/tdhook) observability and intervention.

## Getting Started

```bash
pip install xdrl
```

```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from xdrl import (
    BatchSemantics,
    KeyPresence,
    KeyRole,
    KeySchema,
    ModelRole,
    TensorDictSchema,
    validate_module,
)

batch = TensorDict({"observation": torch.randn(8, 4)}, batch_size=[8])
policy = TensorDictModule(
    torch.nn.Linear(4, 2),
    in_keys=["observation"],
    out_keys=["action"],
)
policy.role = ModelRole.ACTOR
batch_dims = BatchSemantics(("env",))
policy.input_schema = TensorDictSchema(
    (KeySchema("observation", KeyRole.OBSERVATION, KeyPresence.REQUIRED),),
    batch_dims,
)
policy.output_schema = TensorDictSchema(
    (KeySchema("action", KeyRole.ACTION, KeyPresence.PRODUCED),),
    batch_dims,
)

result = validate_module(policy, batch)
assert result["action"].shape == (8, 2)
```

## Development

This project uses [`uv`](https://docs.astral.sh/uv/) to manage Python
dependencies and [`just`](https://github.com/casey/just) to run the
conformance and documentation gates.

## Documentation

- [Getting Started](https://xdrl.readthedocs.io/en/latest/start.html)
- [Tutorials](https://xdrl.readthedocs.io/en/latest/tutorials.html)
- [API Reference](https://xdrl.readthedocs.io/en/latest/api/index.html)
- [About](https://xdrl.readthedocs.io/en/latest/about.html)

## License
`xdrl` is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
