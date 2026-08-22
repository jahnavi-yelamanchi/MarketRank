.PHONY: install lint test run

install:
	uv sync --all-groups

lint:
	uv run ruff check .

test:
	uv run pytest

benchmark:
	uv run python scripts/benchmark_serving.py --iterations 250 --workers 8

run:
	uv run uvicorn marketrank.api:app --reload
