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

Refresh the development lock from a clean branch, review the resolved
revisions, and run the normal gates:

```bash
uv lock
just checks
just tests
just docs
```

The lockfile records the tested development environment; it is not a second
runtime compatibility API. Keep a failed dependency proposal intact for
diagnosis rather than weakening package constraints to make it pass.

## Branches

Make a branch before making a pull request to `develop`.

## Scientific reproduction notebooks

- One primary notebook per paper in `docs/source/reproductions/`; do not reuse tutorials.
- Header: paper/code links, revisions, execution mode, assets, claim limits.
- Report smoke, artifacts, reference agreement, and paper claims separately.
- Keep paper-specific code beside the notebook and large assets outside Git.
- Publish it from `docs/source/tutorials.rst` only after it uses the current
  public API and passes `just docs`.
