.PHONY: install lint test run

install:
	uv sync --all-groups

lint:
	uv run ruff check .

test:
	uv run pytest

run:
	uv run uvicorn marketrank.api:app --reload

