.PHONY: install test lint demo serve

install:
	python -m pip install -e ".[dev,serve]"

test:
	python -m pytest -q

lint:
	ruff check .

demo:
	python -m vla_forge demo-planner

serve:
	uvicorn vla_forge.api:app --host 127.0.0.1 --port 8000
