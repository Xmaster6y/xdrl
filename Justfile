install:
	uv run pre-commit install
	uv sync

checks:
	uv run pre-commit run --all-files

check-dependency-snapshot:
	uv run python scripts/dependency_snapshot.py check

sync-dependency-snapshot:
	uv run python scripts/dependency_snapshot.py sync

refresh-dependencies:
	uv run python scripts/dependency_snapshot.py refresh
	just dependency-gates

dependency-gates:
	just check-dependency-snapshot
	just checks
	just tests
	just docs

test-assets:
	@echo "No test assets to resolve"

tests:
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v

test-unit:
	uv run pytest tests/unit -s -v

test-upstream-compatibility:
	uv run pytest tests/upstream_compatibility -s -v

test-integration:
	uv run pytest tests/integration -s -v

test-behavioural-parity:
	uv run pytest tests/behavioural_parity -s -v

wandb-sync:
	uv run --no-sync wandb sync --sync-all

launch cluster script *args:
    sbatch launch/{{cluster}}/{{script}}.sh {{args}}

run script *args:
    uv run -m scripts.{{script}} {{args}}

docs:
	cd docs && uv run --group docs make html SPHINXOPTS="-W --keep-going"
