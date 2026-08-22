PORT ?= 8001

.PHONY: install lint test run

install:
	uv sync --all-groups

lint:
	uv run ruff check .

test:
	uv run pytest

benchmark:
	uv run python scripts/benchmark_serving.py --iterations 250 --workers 8

train-demo:
	uv run python scripts/train_demo.py --requests 120 --seed 7

run:
	uv run uvicorn marketrank.api:app --reload --port $(PORT)
