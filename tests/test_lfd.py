from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from agent_evals.cli import app
from agent_evals.lfd import (
    CycleRecord,
    LfdConfig,
    LfdState,
    evaluate,
    extract_misses,
    load_state,
    render_goal,
    save_state,
)
from agent_evals.models import (
    DimensionScore,
    RunManifest,
    RunResult,
    RunSummary,
    Scorecard,
    Trace,
)

runner = CliRunner()

NOW = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)


def _cycle(n: int, overall: float, cost: float | None = 0.01) -> CycleRecord:
    return CycleRecord(
        n=n,
        utc_timestamp=NOW.isoformat(),
        hypothesis=f"h{n}",
        overall=overall,
        passed=1,
        task_count=1,
        cost_usd=cost,
    )


def _card(task_id: str, *, overall: float, passed: bool, dims: dict[str, tuple[float, str]]) -> Scorecard:
    return Scorecard(
        task_id=task_id,
        category="coding",
        provider="anthropic",
        model="claude-opus-4-8",
        dimensions={
            name: DimensionScore(score=score, weight=1.0, reason=reason)
            for name, (score, reason) in dims.items()
        },
        overall=overall,
        passed=passed,
        cost_usd=0.01,
        latency_s=1.0,
        failed=False,
        trace=Trace(provider="anthropic", model="claude-opus-4-8"),
    )


def _run(cards: list[Scorecard]) -> RunResult:
    overall = sum(card.overall for card in cards) / len(cards)
    summary = RunSummary(
        task_count=len(cards),
        overall_mean=overall,
        passed=sum(card.passed for card in cards),
        failure_rate=0.0,
        total_cost_usd=0.01,
        median_latency_s=1.0,
        by_dimension={},
        by_category={},
    )
    manifest = RunManifest(
        schema_version="2026-05-29.1",
        package_version="0.1.0",
        provider="anthropic",
        model="claude-opus-4-8",
        judge_model="claude-haiku-4-5",
        seed=42,
        tool_layer_version="shared-tools-v1",
        pricing_hash="abc",
        task_hashes={card.task_id: "x" for card in cards},
    )
    return RunResult(
        schema_version="2026-05-29.1",
        manifest=manifest,
        summary=summary,
        scorecards=cards,
        cassette={},
    )


# --- state / verdict unit tests -------------------------------------------


def test_stalled_for_counts_cycles_since_last_best():
    state = LfdState(cycles=[_cycle(1, 0.5), _cycle(2, 0.6), _cycle(3, 0.6), _cycle(4, 0.55)])
    assert state.stalled_for == 2
    state.cycles.append(_cycle(5, 0.7))
    assert state.stalled_for == 0


def test_verdict_target_met_uses_latest_cycle():
    cfg = LfdConfig(target_overall=0.6)
    state = LfdState(cycles=[_cycle(1, 0.7), _cycle(2, 0.5)])
    # Best is 0.7 but the *current* system scores 0.5 — target is not met.
    assert not evaluate(state, cfg, NOW).target_met
    state.cycles.append(_cycle(3, 0.65))
    assert evaluate(state, cfg, NOW).target_met


def test_wallclock_budget_exhausts():
    cfg = LfdConfig(wallclock_budget_h=1.0)
    state = LfdState(started_at=(NOW - timedelta(hours=2)).isoformat(), cycles=[_cycle(1, 0.5)])
    verdict = evaluate(state, cfg, NOW)
    assert any("wall-clock" in reason for reason in verdict.exhausted)


def test_cost_and_cycle_budgets_exhaust():
    cfg = LfdConfig(cost_budget_usd=0.05, max_cycles=3)
    state = LfdState(cycles=[_cycle(n, 0.5, cost=0.02) for n in range(1, 4)])
    verdict = evaluate(state, cfg, NOW)
    assert any("spend" in reason for reason in verdict.exhausted)
    assert any("cycles" in reason for reason in verdict.exhausted)


def test_no_budgets_never_exhaust():
    state = LfdState(started_at=(NOW - timedelta(hours=100)).isoformat(), cycles=[_cycle(1, 0.1)])
    assert evaluate(state, LfdConfig(), NOW).exhausted == []


def test_stall_flag_respects_threshold():
    cfg = LfdConfig(stall_cycles=3)
    state = LfdState(cycles=[_cycle(1, 0.5), _cycle(2, 0.5), _cycle(3, 0.5)])
    assert not evaluate(state, cfg, NOW).stalled  # 2 cycles since best
    state.cycles.append(_cycle(4, 0.5))
    assert evaluate(state, cfg, NOW).stalled


def test_state_roundtrip(tmp_path: Path):
    state = LfdState(started_at=NOW.isoformat(), cycles=[_cycle(1, 0.5)])
    path = tmp_path / "nested" / "state.json"
    save_state(state, path)
    assert load_state(path) == state
    assert load_state(tmp_path / "missing.json") == LfdState()


# --- blinded misses ---------------------------------------------------------


def test_extract_misses_skips_passing_tasks_and_flags_weak_dims():
    run = _run(
        [
            _card("ok", overall=0.9, passed=True, dims={"completion": (0.9, "")}),
            _card(
                "bad",
                overall=0.4,
                passed=False,
                dims={"completion": (0.3, "missed the ARR figure"), "tool": (0.95, "")},
            ),
        ]
    )
    misses = extract_misses(run)
    assert [miss.task_id for miss in misses] == ["bad"]
    assert misses[0].weak_dimensions == {"completion": 0.3}
    # Judge rationales are carried for opt-in display, never mixed into scores.
    assert misses[0].reasons == {"completion": "missed the ARR figure"}


# --- goal rendering ---------------------------------------------------------


def test_render_goal_includes_target_and_budgets():
    goal = render_goal(LfdConfig(target_overall=0.9, wallclock_budget_h=4, cost_budget_usd=25, max_cycles=50))
    assert "overall >= 0.9" in goal
    assert "4 hours" in goal
    assert "$25" in goal
    assert "50" in goal


def test_render_goal_flags_unset_budgets():
    goal = render_goal(LfdConfig())
    assert goal.count("UNSET") == 2


# --- CLI end-to-end (stub agent, no API keys) -------------------------------

PLUGIN_SRC = '''
import os
from agent_evals.models import Trace
from agent_evals.plugins import AgentPlugin
from agent_evals.tools import build_default_registry


class StubAdapter:
    provider = "stub-agent"
    model = "stub-1"

    def run_task(self, task, registry, ctx):
        if os.environ.get("STUB_QUALITY") == "low":
            output = ""
        else:
            output = " ".join(task.expected_outcomes)
        return Trace(
            provider="stub-agent",
            model="stub-1",
            final_output=output,
            finished_reason="completed",
        )


def build(cfg):
    return AgentPlugin(adapter=StubAdapter(), registry=build_default_registry())
'''


def _write_project(tmp_path: Path, lfd_section: str) -> Path:
    (tmp_path / "stub_plugin.py").write_text(PLUGIN_SRC, encoding="utf-8")
    config = tmp_path / "agentevals.toml"
    config.write_text(
        f"""
[project]
baseline = "baseline.json"
record   = "replay"
task_ids = ["company_research_001"]

[agent]
plugin = "stub_plugin:build"

[lfd]
{lfd_section}
""",
        encoding="utf-8",
    )
    return config


def test_score_below_target_exits_1_and_logs_cycle(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("STUB_QUALITY", "low")
    config = _write_project(tmp_path, 'target_overall = 0.99\nstate = "state.json"')

    res = runner.invoke(
        app,
        ["lfd", "score", "--hypothesis", "baseline attempt", "--config", str(config), "--out", str(tmp_path / "results")],
    )
    assert res.exit_code == 1, res.output
    state = load_state(tmp_path / "state.json")
    assert len(state.cycles) == 1
    assert state.cycles[0].hypothesis == "baseline attempt"
    assert state.started_at is not None


def test_score_is_blinded_by_default(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("STUB_QUALITY", "low")
    config = _write_project(tmp_path, 'target_overall = 0.99\nstate = "state.json"')
    args = ["lfd", "score", "--hypothesis", "h", "--config", str(config), "--out", str(tmp_path / "results")]

    blinded = runner.invoke(app, args)
    assert "Judge reasons" not in blinded.output
    unblinded = runner.invoke(app, args + ["--show-reasons"])
    assert "Judge reasons" in unblinded.output


def test_score_target_met_exits_0(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("STUB_QUALITY", raising=False)
    config = _write_project(tmp_path, 'target_overall = 0.0\nstate = "state.json"')

    res = runner.invoke(
        app,
        ["lfd", "score", "--hypothesis", "good agent", "--config", str(config), "--out", str(tmp_path / "results")],
    )
    assert res.exit_code == 0, res.output
    assert "TARGET REACHED" in res.output


def test_score_refuses_to_run_past_cycle_budget(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("STUB_QUALITY", "low")
    config = _write_project(tmp_path, 'target_overall = 0.99\nmax_cycles = 1\nstate = "state.json"')
    args = ["lfd", "score", "--hypothesis", "h", "--config", str(config), "--out", str(tmp_path / "results")]

    first = runner.invoke(app, args)
    assert first.exit_code == 3, first.output  # budget hit at end of the cycle
    second = runner.invoke(app, args)
    assert second.exit_code == 3, second.output  # blocked before spending anything
    assert len(load_state(tmp_path / "state.json").cycles) == 1


def test_status_and_log_report_cycles(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("STUB_QUALITY", "low")
    config = _write_project(tmp_path, 'target_overall = 0.99\nstate = "state.json"')
    runner.invoke(
        app,
        ["lfd", "score", "--hypothesis", "first idea", "--config", str(config), "--out", str(tmp_path / "results")],
    )

    status = runner.invoke(app, ["lfd", "status", "--config", str(config)])
    assert status.exit_code == 0, status.output
    assert "target=0.99" in status.output
    assert "Instruments" in status.output

    log = runner.invoke(app, ["lfd", "log", "--config", str(config)])
    assert log.exit_code == 0, log.output
    assert "first idea" in log.output


def test_status_with_no_cycles(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _write_project(tmp_path, 'state = "state.json"')
    res = runner.invoke(app, ["lfd", "status", "--config", str(config)])
    assert res.exit_code == 0, res.output
    assert "No cycles yet" in res.output


def test_init_writes_goal_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _write_project(
        tmp_path, 'target_overall = 0.9\nwallclock_budget_h = 4\ncost_budget_usd = 25\ngoal = "GOAL.md"'
    )

    res = runner.invoke(app, ["lfd", "init", "--config", str(config)])
    assert res.exit_code == 0, res.output
    goal = (tmp_path / "GOAL.md").read_text(encoding="utf-8")
    assert "overall >= 0.9" in goal

    again = runner.invoke(app, ["lfd", "init", "--config", str(config)])
    assert again.exit_code == 2, again.output
    forced = runner.invoke(app, ["lfd", "init", "--config", str(config), "--force"])
    assert forced.exit_code == 0, forced.output
