.PHONY: setup test smoke smoke-openai smoke-anthropic

setup:
	uv sync --extra dev

test:
	uv run pytest

smoke-anthropic:
	uv run agent-evals run --provider anthropic --tasks company_research_001,coding_001 --judge-model claude-haiku-4-5 --out results/

smoke-openai:
	uv run agent-evals run --provider openai --tasks company_research_001,coding_001 --judge-model claude-haiku-4-5 --out results/

smoke: smoke-anthropic
