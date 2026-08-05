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

Typed model interactions for [TorchRL](https://github.com/pytorch/rl), with
[TDHook](https://github.com/Xmaster6y/tdhook) observability and intervention.

## Getting Started

`xdrl` keeps TensorDict data and TorchRL execution native. It adds explicit
schemas and execution context around model calls, then runs TDHook v0.2
workflows through that interaction with exception-safe cleanup and model-pass
evidence.

```python
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow
from xdrl import TDHookWorkflowRunner

# `interaction` is a RuntimeInteractionContext declaring the policy role,
# TensorDict schemas, batch semantics, and evaluation/collection phase.
workflow = Workflow(
    ActivationCaching("module.0", cache_key=("activations", "encoder"))
)
execution = TDHookWorkflowRunner(interaction).run(
    workflow, batch, code_revision="your-git-revision"
)
encoder_activations = execution.data["activations", "encoder"]
```

See the [complete quickstart](https://xdrl.readthedocs.io/en/latest/start.html),
[architecture](https://xdrl.readthedocs.io/en/latest/architecture.html), and
[compatibility contract](https://xdrl.readthedocs.io/en/latest/compatibility.html).

The supported boundary is currently local, synchronous TensorDict module
execution. Compiled, remote, distributed, and worker-copied policies are not
silently treated as supported.

Development tracks the latest TensorDict and TorchRL `main` branches. The
lockfile records the exact revisions exercised by CI; see the
[compatibility contract](https://xdrl.readthedocs.io/en/latest/compatibility.html#development-dependency-policy).

## Development

This project uses [`uv`](https://docs.astral.sh/uv/) to manage Python
dependencies and [`just`](https://github.com/casey/just) to run the
conformance and documentation gates.

## Documentation

See the full documentation at <https://xdrl.readthedocs.io>.

## License
`xdrl` is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
