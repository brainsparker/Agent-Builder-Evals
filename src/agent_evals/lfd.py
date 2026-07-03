"""Loss Function Development (LFD): run the eval suite as an optimization target.

Where `check` is a gate (did the agent get worse?), `lfd` is a loss function
(is the agent at the target yet?) for an outer optimization loop — a human or
an autonomous agent repeatedly changing the system and re-scoring until the
target is met or a budget runs out.

The loss function is more than the metric. It has four parts, each of which
maps onto something concrete here:

- **Target**: `target_overall`, scored blind — `lfd score` reports scores and
  a per-item miss list but never the answer key (reference answers, expected
  outcomes, and judge rationales stay hidden by default).
- **Constraints**: wall-clock, dollar, and cycle budgets. A blown budget stops
  the loop with a distinct exit code instead of letting the optimizer grind.
- **Instruments**: every constraint is inspectable (`lfd status`) — elapsed
  vs. budget, cumulative spend vs. budget, tokens per cycle. You can't
  optimize (or respect) what you can't see.
- **Forced entropy**: every cycle records a hypothesis, and a stall (no new
  best score for `stall_cycles` cycles) demands a qualitatively different next
  move rather than the same knob turned harder.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from agent_evals.models import RunResult

# A dimension scoring below this on a missed task is surfaced in the miss list.
DIM_MISS_THRESHOLD = 0.7

# Exit codes for `lfd score` (0/2 match `check`: success / setup error).
EXIT_TARGET_MET = 0
EXIT_BELOW_TARGET = 1
EXIT_CONFIG_ERROR = 2
EXIT_BUDGET_EXHAUSTED = 3


class LfdConfig(BaseModel):
    """The `[lfd]` section of agentevals.toml.

    Budgets left at ``None`` are unlimited — set them. An optimizer with no
    wall-clock or dollar ceiling will happily grind for hours on a 0.1% gain.
    """

    target_overall: float = Field(default=0.95, ge=0, le=1)
    wallclock_budget_h: float | None = None
    cost_budget_usd: float | None = None
    max_cycles: int | None = None
    stall_cycles: int = 3
    state: Path = Path("results/lfd_state.json")
    goal: Path = Path("GOAL.md")


class CycleRecord(BaseModel):
    """One optimization cycle: what was tried, what it scored, what it cost."""

    n: int
    utc_timestamp: str
    hypothesis: str
    overall: float
    passed: int
    task_count: int
    by_dimension: dict[str, float] = Field(default_factory=dict)
    cost_usd: float | None = None
    tokens: int = 0
    duration_s: float = 0.0
    run_file: str | None = None


class LfdState(BaseModel):
    """The iteration log, persisted across cycles (and context compactions)."""

    started_at: str | None = None
    cycles: list[CycleRecord] = Field(default_factory=list)

    @property
    def best_overall(self) -> float | None:
        if not self.cycles:
            return None
        return max(cycle.overall for cycle in self.cycles)

    @property
    def spent_usd(self) -> float:
        return sum(cycle.cost_usd or 0.0 for cycle in self.cycles)

    @property
    def stalled_for(self) -> int:
        """Consecutive trailing cycles since the best score last improved."""
        best: float | None = None
        since = 0
        for cycle in self.cycles:
            if best is None or cycle.overall > best:
                best = cycle.overall
                since = 0
            else:
                since += 1
        return since

    def elapsed_h(self, now: datetime) -> float | None:
        if self.started_at is None:
            return None
        started = datetime.fromisoformat(self.started_at)
        return (now - started).total_seconds() / 3600


class Miss(BaseModel):
    """A blinded miss: which task fell short and on what dimensions.

    Deliberately excludes the answer key. ``reasons`` (judge rationales) are
    carried for opt-in display only — they can leak eval-shaped hints that an
    optimizer will memorize one keyword at a time.
    """

    task_id: str
    category: str
    overall: float
    failed: bool
    weak_dimensions: dict[str, float] = Field(default_factory=dict)
    reasons: dict[str, str] = Field(default_factory=dict)


class Verdict(BaseModel):
    target_met: bool
    exhausted: list[str] = Field(default_factory=list)
    stalled: bool = False


def load_state(path: Path) -> LfdState:
    if not path.exists():
        return LfdState()
    return LfdState.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_state(state: LfdState, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return path


def record_cycle(
    state: LfdState,
    *,
    hypothesis: str,
    run: RunResult,
    run_file: Path | None,
    duration_s: float,
    now: datetime,
) -> CycleRecord:
    if state.started_at is None:
        state.started_at = now.isoformat()
    tokens = sum(
        card.trace.usage.input_tokens + card.trace.usage.output_tokens
        for card in run.scorecards
    )
    cycle = CycleRecord(
        n=len(state.cycles) + 1,
        utc_timestamp=now.isoformat(),
        hypothesis=hypothesis,
        overall=run.summary.overall_mean,
        passed=run.summary.passed,
        task_count=run.summary.task_count,
        by_dimension=dict(run.summary.by_dimension),
        cost_usd=run.summary.total_cost_usd,
        tokens=tokens,
        duration_s=duration_s,
        run_file=str(run_file) if run_file else None,
    )
    state.cycles.append(cycle)
    return cycle


def evaluate(state: LfdState, cfg: LfdConfig, now: datetime | None = None) -> Verdict:
    """Where does the loop stand: at target, out of budget, or stalled?"""
    now = now or datetime.now(UTC)
    exhausted: list[str] = []

    elapsed = state.elapsed_h(now)
    if cfg.wallclock_budget_h is not None and elapsed is not None and elapsed >= cfg.wallclock_budget_h:
        exhausted.append(f"wall-clock: {elapsed:.1f}h elapsed >= {cfg.wallclock_budget_h:.1f}h budget")
    if cfg.cost_budget_usd is not None and state.spent_usd >= cfg.cost_budget_usd:
        exhausted.append(f"spend: ${state.spent_usd:.2f} >= ${cfg.cost_budget_usd:.2f} budget")
    if cfg.max_cycles is not None and len(state.cycles) >= cfg.max_cycles:
        exhausted.append(f"cycles: {len(state.cycles)} >= {cfg.max_cycles} budget")

    latest = state.cycles[-1].overall if state.cycles else None
    return Verdict(
        target_met=latest is not None and latest >= cfg.target_overall,
        exhausted=exhausted,
        stalled=state.stalled_for >= cfg.stall_cycles,
    )


def extract_misses(run: RunResult) -> list[Miss]:
    """Per-item miss list from a scored run, sans answer key.

    Reads only scorecard scores — never the task definitions — so reference
    answers and expected outcomes cannot leak through this path.
    """
    misses: list[Miss] = []
    for card in run.scorecards:
        if card.passed and not card.failed:
            continue
        weak = {
            name: dim.score
            for name, dim in sorted(card.dimensions.items(), key=lambda kv: kv[1].score)
            if dim.score < DIM_MISS_THRESHOLD
        }
        reasons = {name: card.dimensions[name].reason for name in weak if card.dimensions[name].reason}
        misses.append(
            Miss(
                task_id=card.task_id,
                category=card.category.value,
                overall=card.overall,
                failed=card.failed,
                weak_dimensions=weak,
                reasons=reasons,
            )
        )
    return misses


def _budget_line(label: str, value: str) -> str:
    return f"- **{label}**: {value}"


def render_goal(cfg: LfdConfig) -> str:
    """Render the /goal prompt for an optimization loop from the loss function."""
    wallclock = f"{cfg.wallclock_budget_h:g} hours" if cfg.wallclock_budget_h is not None else "UNSET — set `wallclock_budget_h` in [lfd]"
    dollars = f"${cfg.cost_budget_usd:g}" if cfg.cost_budget_usd is not None else "UNSET — set `cost_budget_usd` in [lfd]"
    cycles = str(cfg.max_cycles) if cfg.max_cycles is not None else "unlimited"
    return f"""# /goal — optimize the agent until the eval says you're done

Improve the agent under test (see `agentevals.toml`) until
`agent-evals lfd score` reports **overall >= {cfg.target_overall:g}**, within budget.

## Target

- The only score that counts is the one printed by
  `agent-evals lfd score --hypothesis "..."`. Run it after every change.
- The eval is **blind**. Do not read the eval task YAML, reference answers,
  or expected outcomes. You get scores and a per-item miss list — nothing else.
- Exit codes: `0` target met (stop, you're done), `1` below target (keep
  going), `3` budget exhausted (stop and report), `2` setup error (fix setup).

## Constraints

{_budget_line("Wall-clock budget", wallclock)}
{_budget_line("Dollar budget", dollars)}
{_budget_line("Cycle budget", cycles)}
- Improve the agent (prompt, tools, logic) — never the scorer, the eval
  tasks, or this harness. A higher number from a gamed metric is a failure.
- No hardcoding eval-shaped artifacts: no keyword lists, seed data, or
  special cases that exist only to satisfy specific eval items.

## Instruments

- `agent-evals lfd score --hypothesis "..."` — run + score one cycle
  (records it in the iteration log).
- `agent-evals lfd status` — target vs. best/last score, elapsed vs.
  wall-clock budget, spend vs. dollar budget, cycles used, stall state.
- `agent-evals lfd log` — the full iteration log: every cycle's hypothesis,
  score, delta, cost, and duration.

Check `lfd status` between cycles. If a budget is nearly exhausted, prefer
finishing cleanly over starting a new expensive idea.

## Forced entropy

- Every cycle needs a real `--hypothesis`: what you changed and why it should
  move the metric. "Try again" is not a hypothesis.
- **Overfit reflection, every cycle**: am I building a more general solution,
  or memorizing the eval? If memorizing, the next change must remove an
  eval-shaped artifact, not add one.
- **On stall** ({cfg.stall_cycles} cycles without a new best): the next cycle
  must be a qualitatively different approach — a different subsystem,
  strategy, or representation. Not the same knob turned harder.
- Read `agent-evals lfd log` before each cycle so you don't repeat a
  hypothesis that already failed.
"""
