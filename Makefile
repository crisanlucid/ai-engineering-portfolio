.PHONY: install lint fmt type-check test

install:
	uv sync --all-packages

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

type-check:
	uv run mypy packages/ projects/

test:
	uv run pytest

check: lint type-check test
