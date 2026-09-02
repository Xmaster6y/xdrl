install:
	uv run pre-commit install
	uv sync

checks:
	uv run pre-commit run --all-files

test-assets:
	@echo "No test assets to resolve"

tests:
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v

test-unit:
	uv run pytest tests/unit -s -v

test-integration:
	uv run pytest tests/integration -s -v

test-behavioural-parity:
	uv run pytest tests/behavioural_parity -s -v

docs:
	cd docs && uv run --group docs make html SPHINXOPTS="-W --keep-going"
