.PHONY: install lint format test data compare tune ngrams topics

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

data:
	uv run python -m reviews.data

compare:
	uv run python -m reviews.compare

tune:
	uv run python -m reviews.tune

ngrams:
	uv run python -m reviews.ngrams

topics:
	uv run python -m reviews.topics
