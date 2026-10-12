# Common tasks. Run `make` to list them.
.DEFAULT_GOAL := help
.PHONY: help setup assets lint format test check-render check-smolvla expert-eval

help: ## List tasks
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup: ## Install the locked environment, git hooks and robot assets
	uv sync --locked
	uv run pre-commit install
	uv run python scripts/fetch_assets.py

assets: ## Fetch pinned upstream robot assets into third_party/
	uv run python scripts/fetch_assets.py

lint: ## Run all pre-commit checks (ruff, formatting, file hygiene)
	uv run pre-commit run --all-files

format: ## Apply lint fixes and formatting
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run the test suite
	uv run pytest

check-render: ## M0: measure headless rendering and physics throughput
	uv run python scripts/check_render.py

check-smolvla: ## M0: measure SmolVLA memory, training-step time and inference latency
	uv run python scripts/check_smolvla.py

expert-eval: ## M1: evaluate the scripted expert on 100 held-out seeds, with one video
	uv run python scripts/run_expert.py --episodes 100 --seed-start 1000 --videos 1
