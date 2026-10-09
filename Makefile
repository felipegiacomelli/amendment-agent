.PHONY: install lint fmt test check

install:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run black --check .
	uv run mypy

fmt:
	uv run ruff check --fix .
	uv run black .

# Fails below [tool.coverage.report] fail_under, so coverage can only ratchet up.
test:
	uv run pytest --cov --cov-report=term-missing

check: lint test
