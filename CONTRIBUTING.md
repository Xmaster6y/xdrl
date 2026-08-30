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
a tutorial notebook. Follow the compact header and status contract on the
[Reproductions page](https://xdrl.readthedocs.io/en/latest/reproductions/).

Keep paper-specific code beside the notebook; reusable library code belongs in
`src/`. Keep large or licensed assets outside Git and record their source,
revision, checksum, and local path in the notebook. Link durable artifacts from
the gallery, then run `just docs`; documentation rendering is not experiment
evidence.
