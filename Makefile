.PHONY: lint

lint:
	uv run ruff check --fix .
	uv run ruff format .