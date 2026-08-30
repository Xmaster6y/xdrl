# How to Contribute?

## Guidelines

The project dependencies are managed using `uv`, see their installation [guide](https://docs.astral.sh/uv/).

Additionally, to make your life easier, install `just` to use the shortcut commands.

## Dev Install

Install the dependencies and the pre-commit hooks:

```bash
just install
```

To run the checks (`pre-commit` checks):

```bash
just checks
```

To run the tests (using `pytest`):

```bash
just tests
```

## Refreshing development dependencies

`uv.lock` is the exact compatibility snapshot; a branch name or a successful
installation is not support evidence. Run the single refresh workflow from a
clean branch:

```bash
just refresh-dependencies
```

The command resolves the configured TDHook and TorchRL development branches,
refreshes TDHook, TensorDict, PyTorch, and TorchRL, synchronises
`SUPPORTED_DEPENDENCIES`, `SUPPORTED_GIT_REVISIONS`, and the documented matrix,
prints an old/new version and Git-revision table, then runs formatting, unit,
upstream-compatibility, integration, behavioural-parity, coverage, and
documentation gates. Keep a failed proposal intact for diagnosis; do not
replace a failing revision or widen the generated declarations without passing
the gates.

CI runs `just check-dependency-snapshot` and fails when `uv.lock`, the runtime
declarations, or this documentation disagree. The scheduled/manual
`Dependency snapshot refresh` action uploads the complete patch even when a gate
fails, so the proposal remains auditable rather than becoming a support claim.

PyTorch currently resolves from its stable index for the supported Python
3.11--3.13 CI matrix. If a platform needs the PyTorch nightly index, declare it
with a mutually exclusive environment marker in `tool.uv.sources`, document the
same local-development marker here, and verify every supported Python/platform
fork in the generated universal lock before accepting the refresh.

## Branches

Make a branch before making a pull request to `develop`.

## Scientific reproduction notebooks

Scientific reproductions are separate from task-oriented tutorials. Give each
paper exactly one primary notebook under `docs/source/reproductions/`, add its
gallery entry to `docs/source/reproductions/index.rst`, and do not move or reuse
a notebook from `docs/source/notebooks/` as the deliverable. Follow the notebook
header and evidence-status contract documented on the
[Reproductions page](https://xdrl.readthedocs.io/en/latest/reproductions/).

Keep paper-specific orchestration and analysis in the primary notebook or in a
small support module beside it. Changes to reusable XDRL behavior belong under
`src/` and require their own focused feature or design issue. Small,
redistributable fixtures may be committed beside the notebook. Models,
datasets, environment installations, and other large or externally licensed
assets must stay outside Git: document their source, license, expected local
path, version or revision, and checksum in the notebook instead. Publish durable
result artifacts at a stable external location and link them from the notebook
and gallery; the ignored local `outputs/` directory is only a working area.

Run `just docs` after adding a reproduction. The build renders notebooks without
executing them, so a successful documentation build is not smoke-execution or
scientific-result evidence.
