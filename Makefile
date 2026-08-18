.PHONY: test check fmt train serve

test:
	uv run pytest -q

check:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .

train:
	uv run python matcher.py

serve:
	uv run uvicorn api:app --reload