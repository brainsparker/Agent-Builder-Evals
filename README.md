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

## Optimization loops: loss-function development (`lfd`)

`check` is a gate — it tells you when the agent got *worse*. `agent-evals lfd`
turns the same eval suite into a **loss function** for an outer optimization
loop: point a coding agent (or yourself) at a target score and iterate until
the target is met or a budget runs out.

```bash
agent-evals lfd init                                # write GOAL.md — the /goal prompt for the loop
agent-evals lfd score --hypothesis "cache retries"  # run one cycle: execute, score blind, log it
agent-evals lfd status                              # target vs. score, budgets vs. burn, stall state
agent-evals lfd log                                 # every cycle's hypothesis, score, Δ, cost, time
```

An optimizer takes every shortcut you leave open, so the loss function is more
than the metric — `[lfd]` in `agentevals.toml` encodes all four parts:

- **Target** — `target_overall`, scored **blind**: `lfd score` reports scores
  and a per-item miss list (task, weak dimensions) but never the answer key.
  Judge rationales are hidden unless you pass `--show-reasons`, because an
  optimizer will memorize them one keyword at a time.
- **Constraints** — `wallclock_budget_h`, `cost_budget_usd`, `max_cycles` are
  hard stops, not vibes: a blown budget refuses to run and exits `3`.
- **Instruments** — every constraint is inspectable (`lfd status`): elapsed
  vs. time budget, cumulative spend vs. dollar budget, tokens and seconds per
  cycle. You can't optimize what you can't see.
- **Forced entropy** — every cycle requires a `--hypothesis`, the iteration
  log survives context compactions, and a stall (`stall_cycles` without a new
  best) demands a qualitatively different next move, not the same knob turned
  harder.

Exit codes make the loop scriptable: `0` target met, `1` below target (keep
going), `3` budget exhausted, `2` setup error. A minimal unattended loop:

```bash
agent-evals lfd init && cat GOAL.md   # hand this to your agent as its /goal
# the agent then repeats: change code → agent-evals lfd score --hypothesis "…"
# until exit code 0 (done) or 3 (out of budget)
```

Keep the eval tasks out of the optimizer's reach (separate directory, not in
its sandbox): a blinded score plus a wide eval set is what makes the only
profitable direction *actually getting better at the task*.

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
