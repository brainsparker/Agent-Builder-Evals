# Agent Builder Evals

**pytest for your AI agent.** Run it on every change to catch regressions in quality, cost, and latency before they ship — and prove it with a replayable JSON trace for every result.

Other benchmarks hand you a number you have to trust. Agent Builder Evals hands you a **gate** (`agent-evals check` exits nonzero when your agent gets worse) plus the full, version-stamped decision tree behind every score, so anyone can re-run the evidence.

```bash
agent-evals check          # run your agent, fail CI if it regressed vs the baseline
```

## Quick start (CI gate)

```bash
uv sync --extra dev
cp .env.example .env        # add your provider key(s)
```

1. Describe your agent and tolerances in `agentevals.toml` (an annotated example ships in the repo root).
2. Pin a baseline from a known-good state:
   ```bash
   agent-evals check --update-baseline
   ```
3. On every change, gate against it:
   ```bash
   agent-evals check        # exit 1 if overall/cost/latency/any dimension regressed past tolerance
   ```

Output is a per-metric diff table; the candidate is **always executed fresh** (never replayed), so the gate can actually fail.

### `agentevals.toml`

```toml
[project]
tasks_dir = "evals/tasks"     # your own task YAML (omit to use the built-in suite)
baseline  = "results/baseline.json"
record    = "live"

[agent]
provider      = "anthropic"
model         = "claude-opus-4-8"
# system_prompt = "You are Acme's support agent..."   # test your prompt, no code
# plugin        = "myapp.evals:build"                  # or bring your own agent + tools

[tolerances]
overall_drop         = 0.02   # fail if overall score drops > 2 points
cost_increase_pct    = 0.15   # fail if cost rises > 15%
latency_increase_pct = 0.20
[tolerances.per_dimension_drop]
completion = 0.03
citation   = 0.05
```

### In CI

A ready-to-use GitHub Action lives at `.github/workflows/agent-evals.yml`: it runs `agent-evals check` on every PR, fails the build on a regression, and uploads the full `results/` traces as an artifact.

## Bring your own agent & tasks

- **Your tasks:** point `tasks_dir` at a folder of task YAML (same schema as the built-ins). Built-in tasks keep working when omitted.
- **Your prompt:** set `agent.system_prompt` to benchmark your own instructions against the built-in tools — no code required.
- **Your agent + tools:** set `agent.plugin = "module.path:build"`. The factory returns an `AgentPlugin(adapter, registry)`. Your adapter only needs to satisfy the tiny `AgentAdapter` protocol — `provider`, `model`, and `run_task(task, registry, ctx) -> Trace` — so anything that produces a trace is scored and gated through the same path as the built-ins. Custom tools get free record/replay because the cassette is keyed by `{name, args}`.

  ```python
  from agent_evals.plugins import AgentPlugin
  from agent_evals.tools import build_default_registry

  def build(cfg):
      return AgentPlugin(adapter=MyAgentAdapter(model=cfg.model), registry=build_default_registry())
  ```

  > Plugin code is imported into the process — only point at code you trust. HTTP/subprocess agents and plugin-supplied scorers are planned.

## Also: provider comparison

The same harness compares complete agent systems across providers apples-to-apples, since both receive the **same tool specs and the same local tool code** — the adapters only translate a shared `ToolSpec` into each SDK's shape.

```bash
uv run agent-evals run --provider anthropic --model claude-opus-4-8 --tasks company_research_001 --out results/
uv run agent-evals run --provider openai    --model gpt-5.1        --tasks company_research_001 --out results/
uv run agent-evals show    results/run_*.json
uv run agent-evals compare results/run_a.json results/run_b.json
uv run agent-evals replay  results/run_a.json     # rescore a run without re-calling tools
uv run agent-evals list-tasks
```

Shared tool layer: `web_search`, `fetch_url`, `code_exec`, `support_lookup`.

## Tests

```bash
make test
```

Default tests are mocked and need no API keys. Live runs require provider keys and a You.com key for web search.
