from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_evals.cli import app
from agent_evals.gate import Tolerances, evaluate_gate
from agent_evals.lfd import extract_misses
from agent_evals.models import (
    DimensionScore,
    RunManifest,
    RunResult,
    Scorecard,
    Task,
    Trace,
)
from agent_evals.results_io import read_result, write_result_to
from agent_evals.runner import execute_run, replay_result
from agent_evals.scorers.aggregate import summarize, task_pass_counts
from agent_evals.stats import paired_difference, pass_at_k, pass_pow_k, standard_error
from agent_evals.tools import build_default_registry

runner = CliRunner()


def _card(task_id: str, overall: float, *, trial: int = 1, cost=0.01, failed=False) -> Scorecard:
    return Scorecard(
        task_id=task_id,
        category="coding",
        provider="stub",
        model="stub-1",
        dimensions={"completion": DimensionScore(score=overall, weight=1.0, reason="because")},
        overall=overall,
        passed=overall >= 0.7,
        cost_usd=cost,
        latency_s=1.0,
        failed=failed,
        trace=Trace(provider="stub", model="stub-1"),
        trial=trial,
    )


def _run(scores: dict[str, list[float]], **card_kwargs) -> RunResult:
    cards = [
        _card(task_id, overall, trial=n, **card_kwargs)
        for task_id, overalls in scores.items()
        for n, overall in enumerate(overalls, start=1)
    ]
    summary = summarize(cards)
    manifest = RunManifest(
        schema_version="2026-06-09.1",
        package_version="0.2.0",
        provider="stub",
        model="stub-1",
        judge_model="claude-haiku-4-5",
        seed=42,
        tool_layer_version="shared-tools-v1",
        pricing_hash="abc",
        task_hashes={task_id: "x" for task_id in scores},
        trials=summary.trials,
    )
    return RunResult(
        schema_version="2026-06-09.1",
        manifest=manifest,
        summary=summary,
        scorecards=cards,
        cassette={},
    )


# --- estimators -------------------------------------------------------------


def test_pass_estimators_match_the_counting_definition():
    # 7 passes in 10 trials, k = 3: C(10,3) = 120 draws, 1 is all failures,
    # C(7,3) = 35 are all passes.
    assert pass_at_k(7, 10, 3) == pytest.approx(1 - 1 / 120)
    assert pass_pow_k(7, 10, 3) == pytest.approx(35 / 120)


def test_pass_estimators_agree_at_k_1():
    assert pass_at_k(3, 4, 1) == pytest.approx(0.75)
    assert pass_pow_k(3, 4, 1) == pytest.approx(0.75)


def test_pass_estimators_reject_bad_input():
    with pytest.raises(ValueError):
        pass_at_k(4, 3, 1)
    with pytest.raises(ValueError):
        pass_pow_k(1, 3, 4)
    with pytest.raises(ValueError):
        pass_at_k(0, 0, 1)


def test_standard_error_needs_two_values():
    assert standard_error([0.5]) is None
    assert standard_error([0.2, 0.4]) == pytest.approx(0.1)


def test_paired_difference_uses_shared_tasks_only():
    paired = paired_difference({"a": 0.9, "b": 0.5, "x": 0.1}, {"a": 0.8, "b": 0.6, "y": 1.0})
    assert paired is not None
    diff, se, count = paired
    assert count == 2
    assert diff == pytest.approx(0.0)
    assert se == pytest.approx(0.1)
    assert paired_difference({"a": 1.0}, {"b": 1.0}) is None


# --- summary ----------------------------------------------------------------


def test_summary_of_single_trial_run_is_unchanged():
    summary = summarize([_card("a", 0.9), _card("b", 0.5)])
    assert summary.task_count == 2
    assert summary.trials == 1
    assert summary.overall_mean == pytest.approx(0.7)
    assert summary.passed == 1
    assert summary.pass_rate == pytest.approx(0.5)
    assert summary.pass_at_k == summary.pass_pow_k == pytest.approx(0.5)
    assert summary.total_cost_usd == pytest.approx(0.02)


def test_summary_with_trials_reports_rates_per_task():
    run = _run({"steady": [0.9, 0.9, 0.9], "flaky": [0.9, 0.3, 0.9], "broken": [0.2, 0.2, 0.2]})
    summary = run.summary
    assert summary.task_count == 3
    assert summary.trials == 3
    # Tasks that passed on every trial, not trials that passed.
    assert summary.passed == 1
    assert summary.pass_rate == pytest.approx(5 / 9)
    assert summary.pass_at_k == pytest.approx(2 / 3)
    assert summary.pass_pow_k == pytest.approx(1 / 3)
    assert summary.overall_mean == pytest.approx((0.9 + 0.7 + 0.2) / 3)
    assert summary.overall_se is not None and summary.overall_se > 0
    # Real spend across all nine attempts.
    assert summary.total_cost_usd == pytest.approx(0.09)
    assert task_pass_counts(run.scorecards) == {"steady": (3, 3), "flaky": (2, 3), "broken": (0, 3)}


def test_results_written_before_trials_still_load(tmp_path):
    run = _run({"a": [0.9]})
    payload = json.loads(run.model_dump_json())
    for key in ("trials", "pass_rate", "pass_at_k", "pass_pow_k", "overall_se"):
        payload["summary"].pop(key)
    payload["manifest"].pop("trials")
    for card in payload["scorecards"]:
        card.pop("trial")
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    old = read_result(path)
    assert old.summary.trials == 1
    assert old.scorecards[0].trial == 1
    assert not evaluate_gate(old, run, Tolerances()).breached


# --- runner -----------------------------------------------------------------


class _EveryOtherAdapter:
    """Answers correctly on odd attempts and returns nothing on even ones."""

    provider = "stub"
    model = "stub-1"

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.contexts: list[object] = []

    def run_task(self, task, registry, ctx):
        self.contexts.append(ctx)
        self.calls[task.id] = self.calls.get(task.id, 0) + 1
        good = self.calls[task.id] % 2 == 1
        return Trace(
            provider="stub",
            model="stub-1",
            final_output=" ".join(task.expected_outcomes) if good else "",
            finished_reason="completed",
        )


def _task(task_id: str) -> Task:
    return Task(
        id=task_id,
        category="knowledge_retrieval",
        prompt="answer",
        expected_outcomes=["alpha", "beta"],
    )


def test_execute_run_repeats_each_task_with_a_fresh_context():
    adapter = _EveryOtherAdapter()
    run = execute_run(
        adapter=adapter,
        registry=build_default_registry(),
        tasks=[_task("t1"), _task("t2")],
        judge_model="claude-haiku-4-5",
        record_mode="replay",
        seed=42,
        backend=object(),
        trials=3,
    )
    assert [(card.task_id, card.trial) for card in run.scorecards] == [
        ("t1", 1), ("t1", 2), ("t1", 3), ("t2", 1), ("t2", 2), ("t2", 3),
    ]
    assert len({id(ctx) for ctx in adapter.contexts}) == 6
    assert run.manifest.trials == 3
    assert run.summary.task_count == 2
    assert run.summary.trials == 3
    assert run.summary.pass_at_k == pytest.approx(1.0)
    assert run.summary.pass_pow_k == pytest.approx(0.0)


def test_execute_run_rejects_zero_trials():
    with pytest.raises(ValueError):
        execute_run(
            adapter=_EveryOtherAdapter(),
            registry=build_default_registry(),
            tasks=[_task("t1")],
            judge_model="claude-haiku-4-5",
            record_mode="replay",
            seed=42,
            backend=object(),
            trials=0,
        )


def test_replay_keeps_trial_numbers(tmp_path, monkeypatch):
    task = _task("t1")
    run = execute_run(
        adapter=_EveryOtherAdapter(),
        registry=build_default_registry(),
        tasks=[task],
        judge_model="claude-haiku-4-5",
        record_mode="replay",
        seed=42,
        backend=object(),
        trials=2,
    )
    path = write_result_to(run, tmp_path / "run.json")
    monkeypatch.setattr("agent_evals.tasks.loader.load_tasks", lambda *args, **kwargs: [task])
    replayed = read_result(replay_result(path, tmp_path / "replayed"))
    assert [card.trial for card in replayed.scorecards] == [1, 2]
    assert replayed.summary.trials == 2


# --- gate -------------------------------------------------------------------


def test_plain_gate_breaches_and_says_when_it_is_within_noise():
    base = _run({"a": [0.90], "b": [0.80], "c": [0.70], "d": [0.90]})
    cand = _run({"a": [0.95], "b": [0.60], "c": [0.80], "d": [0.85]})  # mean drop 0.025, noisy
    report = evaluate_gate(base, cand, Tolerances())
    row = next(r for r in report.rows if r.name == "overall")
    assert row.breached
    assert row.se is not None and row.se > 0.02
    assert any("run-to-run noise" in w for w in report.warnings)


def test_noise_sigma_lets_a_drop_inside_the_noise_pass():
    base = _run({"a": [0.90], "b": [0.80], "c": [0.70], "d": [0.90]})
    cand = _run({"a": [0.95], "b": [0.60], "c": [0.80], "d": [0.85]})
    report = evaluate_gate(base, cand, Tolerances(noise_sigma=2.0))
    row = next(r for r in report.rows if r.name == "overall")
    assert not row.breached
    assert "within 2 SE of the limit" in row.note
    assert row.tolerance is not None and row.tolerance < -0.02


def test_noise_sigma_still_fails_a_consistent_regression():
    base = _run({"a": [0.90], "b": [0.80], "c": [0.70], "d": [0.90]})
    cand = _run({"a": [0.80], "b": [0.71], "c": [0.60], "d": [0.79]})  # every task drops ~0.10
    report = evaluate_gate(base, cand, Tolerances(noise_sigma=2.0))
    assert next(r for r in report.rows if r.name == "overall").breached
    assert not any("run-to-run noise" in w for w in report.warnings)


def test_noise_sigma_falls_back_to_plain_threshold_without_a_standard_error():
    base = _run({"only": [0.90]})
    cand = _run({"only": [0.80]})
    report = evaluate_gate(base, cand, Tolerances(noise_sigma=2.0))
    row = next(r for r in report.rows if r.name == "overall")
    assert row.breached
    assert row.se is None
    assert "plain threshold" in row.note


def test_noise_sigma_applies_to_dimension_rows():
    base = _run({"a": [0.90], "b": [0.80], "c": [0.70], "d": [0.90]})
    cand = _run({"a": [0.95], "b": [0.60], "c": [0.80], "d": [0.85]})
    plain = evaluate_gate(base, cand, Tolerances(default_dimension_drop=0.02))
    assert next(r for r in plain.rows if r.name == "dim:completion").breached
    lenient = evaluate_gate(base, cand, Tolerances(default_dimension_drop=0.02, noise_sigma=2.0))
    assert not next(r for r in lenient.rows if r.name == "dim:completion").breached


def test_consistency_gate_catches_a_task_that_started_failing_sometimes():
    base = _run({"a": [0.9, 0.9, 0.9], "b": [0.9, 0.9, 0.9]})
    # Mean barely moves, but task b now fails one attempt in three.
    cand = _run({"a": [0.9, 0.9, 0.9], "b": [0.95, 0.95, 0.69]})
    lax = evaluate_gate(base, cand, Tolerances(overall_drop=0.05))
    row = next(r for r in lax.rows if r.name == "pass^3")
    assert not row.breached and "not gated" in row.note
    assert not lax.breached
    assert any("b 2/3" in w for w in lax.warnings)

    strict = evaluate_gate(base, cand, Tolerances(overall_drop=0.05, consistency_drop=0.0))
    row = next(r for r in strict.rows if r.name == "pass^3")
    assert row.baseline == pytest.approx(1.0)
    assert row.candidate == pytest.approx(0.5)
    assert row.breached and strict.breached


def test_consistency_drop_without_trials_warns_instead_of_gating():
    report = evaluate_gate(_run({"a": [0.9]}), _run({"a": [0.9]}), Tolerances(consistency_drop=0.0))
    assert not report.breached
    assert not any(r.name.startswith("pass^") for r in report.rows)
    assert any("consistency_drop is set" in w for w in report.warnings)


def test_trials_mismatch_compares_cost_per_trial_and_warns():
    base = _run({"a": [0.9], "b": [0.9]})
    cand = _run({"a": [0.9, 0.9, 0.9], "b": [0.9, 0.9, 0.9]})  # 3x the spend, same per trial
    report = evaluate_gate(base, cand, Tolerances())
    cost = next(r for r in report.rows if r.name == "cost_usd")
    assert not cost.breached
    assert "per trial" in cost.note
    assert any("trials differ" in w for w in report.warnings)
    assert not report.breached


def test_same_trials_compare_total_cost():
    base = _run({"a": [0.9, 0.9]}, cost=0.01)
    cand = _run({"a": [0.9, 0.9]}, cost=0.02)
    cost = next(r for r in evaluate_gate(base, cand, Tolerances()).rows if r.name == "cost_usd")
    assert cost.breached
    assert cost.baseline == pytest.approx(0.02)
    assert cost.candidate == pytest.approx(0.04)


# --- lfd --------------------------------------------------------------------


def test_misses_list_a_task_once_across_trials():
    run = _run({"steady": [0.9, 0.9], "flaky": [0.9, 0.3]})
    misses = extract_misses(run)
    assert [miss.task_id for miss in misses] == ["flaky"]
    assert misses[0].overall == pytest.approx(0.6)
    assert misses[0].weak_dimensions == {"completion": pytest.approx(0.6)}


# --- cli --------------------------------------------------------------------

PLUGIN_SRC = '''
import os
from agent_evals.models import Trace
from agent_evals.plugins import AgentPlugin
from agent_evals.tools import build_default_registry


class CountingAdapter:
    provider = "stub-agent"
    model = "stub-1"

    def __init__(self):
        self.calls = 0

    def run_task(self, task, registry, ctx):
        self.calls += 1
        # FAIL_EVERY=3 drops the answer on every third attempt.
        every = int(os.environ.get("FAIL_EVERY", "0"))
        good = not every or self.calls % every != 0
        return Trace(
            provider="stub-agent",
            model="stub-1",
            final_output=" ".join(task.expected_outcomes) if good else "",
            finished_reason="completed",
        )


def build(cfg):
    return AgentPlugin(adapter=CountingAdapter(), registry=build_default_registry())
'''


TASK_YAML = """
id: {task_id}
category: knowledge_retrieval
prompt: Name the two things.
expected_outcomes:
  - alpha
  - beta
"""


def _write_project(tmp_path: Path, extra: str = "") -> Path:
    (tmp_path / "trials_plugin.py").write_text(PLUGIN_SRC, encoding="utf-8")
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for task_id in ("first", "second"):
        (tasks / f"{task_id}.yaml").write_text(TASK_YAML.format(task_id=task_id), encoding="utf-8")
    config = tmp_path / "agentevals.toml"
    config.write_text(
        f"""
[project]
baseline = "baseline.json"
record   = "replay"
tasks_dir = "tasks"
trials    = 3

[agent]
plugin = "trials_plugin:build"

[tolerances]
overall_drop = 0.5
{extra}
""",
        encoding="utf-8",
    )
    return config


def test_check_runs_trials_from_config_and_gates_on_consistency(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("FAIL_EVERY", raising=False)
    config = _write_project(tmp_path, "consistency_drop = 0.0")
    out = tmp_path / "results"

    pin = runner.invoke(app, ["check", "--update-baseline", "--config", str(config), "--out", str(out)])
    assert pin.exit_code == 0, pin.output
    assert "3 trials per task" in pin.output
    baseline = read_result(tmp_path / "baseline.json")
    assert len(baseline.scorecards) == 6
    assert baseline.summary.pass_pow_k == pytest.approx(1.0)

    ok = runner.invoke(app, ["check", "--config", str(config), "--out", str(out)])
    assert ok.exit_code == 0, ok.output
    assert "pass^3" in ok.output

    # One attempt in three now fails: the mean stays inside the loose
    # overall tolerance, but no task passes every time any more.
    monkeypatch.setenv("FAIL_EVERY", "3")
    bad = runner.invoke(app, ["check", "--config", str(config), "--out", str(out)])
    assert bad.exit_code == 1, bad.output
    assert "BREACH" in bad.output
    assert "inconsistent tasks" in bad.output


def test_check_trials_flag_overrides_config(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("FAIL_EVERY", raising=False)
    config = _write_project(tmp_path)
    out = tmp_path / "results"
    pin = runner.invoke(
        app, ["check", "--update-baseline", "--trials", "2", "--config", str(config), "--out", str(out)]
    )
    assert pin.exit_code == 0, pin.output
    assert len(read_result(tmp_path / "baseline.json").scorecards) == 4


def test_show_and_compare_report_trials(tmp_path):
    run = _run({"steady": [0.9, 0.9, 0.9], "flaky": [0.9, 0.3, 0.9]})
    path = write_result_to(run, tmp_path / "run.json")
    shown = runner.invoke(app, ["show", str(path)])
    assert shown.exit_code == 0, shown.output
    assert "3 trials per task" in shown.output
    assert "pass^3=50.0%" in shown.output
    compared = runner.invoke(app, ["compare", str(path)])
    assert compared.exit_code == 0, compared.output
    assert "50.0%" in compared.output
