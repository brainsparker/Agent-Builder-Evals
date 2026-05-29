# Agent Builder Evals

Agent Builder Evals is a CLI benchmark suite for evaluating AI agents, not just base models. It compares provider/model/architecture runs across task completion, tool-call accuracy, citation quality, latency, cost, and failure rate.

The MVP is CLI-only. Results are timestamped JSON files in `results/`; `agent-evals show` and `agent-evals compare` are the presentation layer. A future dashboard can read the same JSON contract directly.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

Set keys as needed:

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
YOU_API_KEY=
```

## Commands

```bash
uv run agent-evals list-tasks
uv run agent-evals run --provider anthropic --model claude-opus-4-8 --tasks company_research_001 --out results/
uv run agent-evals run --provider openai --model gpt-5.1 --tasks company_research_001 --out results/
uv run agent-evals show results/run_*.json
uv run agent-evals compare results/run_a.json results/run_b.json
uv run agent-evals replay results/run_a.json
```

## Fairness Model

Both providers receive the same tool specs and execute the same local tool code:

- `web_search`
- `fetch_url`
- `code_exec`
- `support_lookup`

The provider adapters only translate the shared `ToolSpec` into each SDK's expected shape.

## Tests

```bash
make test
```

The default tests are mocked and do not require API keys. Live smoke runs require provider keys and You.com.
