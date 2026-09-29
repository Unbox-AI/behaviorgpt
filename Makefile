.PHONY: lint check test

lint:
	uv run ruff check --fix .
	uv run ruff format .

# what CI runs; format is checked on code only, the docs' aligned code blocks
# are left as written
check:
	uv run ruff check .
	uv run ruff format --check src tests

test:
	uv run pytest
