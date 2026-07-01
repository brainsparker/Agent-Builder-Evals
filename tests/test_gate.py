from __future__ import annotations

from agent_evals.gate import Tolerances, evaluate_gate
from agent_evals.models import (
    RunManifest,
    RunResult,
    RunSummary,
    Scorecard,
    Trace,
)


def _card(task_id: str) -> Scorecard:
    return Scorecard(
        task_id=task_id,
        category="coding",
        provider="anthropic",
        model="claude-opus-4-8",
        dimensions={},
        overall=0.8,
        passed=True,
        cost_usd=0.01,
        latency_s=1.0,
        failed=False,
        trace=Trace(provider="anthropic", model="claude-opus-4-8"),
    )


def _run(
    *,
    overall=0.8,
    cost=0.01,
    latency=1.0,
    failure_rate=0.0,
    by_dimension=None,
    task_ids=("t1",),
    schema_version="2026-05-29.1",
) -> RunResult:
    summary = RunSummary(
        task_count=len(task_ids),
        overall_mean=overall,
        passed=len(task_ids),
        failure_rate=failure_rate,
        total_cost_usd=cost,
        median_latency_s=latency,
        by_dimension=by_dimension or {"completion": overall},
        by_category={"coding": overall},
    )
    manifest = RunManifest(
        schema_version=schema_version,
        package_version="0.1.0",
        provider="anthropic",
        model="claude-opus-4-8",
        judge_model="claude-haiku-4-5",
        seed=42,
        tool_layer_version="shared-tools-v1",
        pricing_hash="abc",
        task_hashes={tid: "x" for tid in task_ids},
    )
    return RunResult(
        schema_version=schema_version,
        manifest=manifest,
        summary=summary,
        scorecards=[_card(tid) for tid in task_ids],
        cassette={},
    )


def test_identical_runs_pass():
    base = _run()
    report = evaluate_gate(base, _run(), Tolerances())
    assert not report.breached


def test_overall_drop_breaches():
    base = _run(overall=0.90)
    cand = _run(overall=0.80)  # -0.10 vs 0.02 tolerance
    report = evaluate_gate(base, cand, Tolerances())
    assert report.breached
    overall_row = next(r for r in report.rows if r.name == "overall")
    assert overall_row.breached


def test_small_overall_drop_within_tolerance():
    base = _run(overall=0.90)
    cand = _run(overall=0.89)  # -0.01 within 0.02
    assert not evaluate_gate(base, cand, Tolerances()).breached


def test_cost_spike_breaches():
    base = _run(cost=0.10)
    cand = _run(cost=0.20)  # +100% vs 15%
    report = evaluate_gate(base, cand, Tolerances())
    assert report.breached
    assert next(r for r in report.rows if r.name == "cost_usd").breached


def test_latency_spike_breaches():
    base = _run(latency=1.0)
    cand = _run(latency=2.0)  # +100% vs 20%
    assert evaluate_gate(base, cand, Tolerances()).breached


def test_per_dimension_drop_breaches():
    base = _run(by_dimension={"completion": 0.9, "citation": 0.9})
    cand = _run(by_dimension={"completion": 0.9, "citation": 0.7})  # citation -0.2 vs 0.05
    tol = Tolerances(per_dimension_drop={"completion": 0.03, "citation": 0.05})
    report = evaluate_gate(base, cand, tol)
    assert report.breached
    assert next(r for r in report.rows if r.name == "dim:citation").breached


def test_ungated_dimension_does_not_breach():
    base = _run(by_dimension={"tool": 0.9})
    cand = _run(by_dimension={"tool": 0.1})  # big drop but no tolerance set
    assert not evaluate_gate(base, cand, Tolerances()).breached


def test_none_cost_is_skipped_not_crash():
    base = _run(cost=None)
    cand = _run(cost=0.05)
    report = evaluate_gate(base, cand, Tolerances())
    cost_row = next(r for r in report.rows if r.name == "cost_usd")
    assert cost_row.breached is False
    assert "skipped" in cost_row.note


def test_task_set_difference_warns():
    base = _run(task_ids=("t1", "t2"))
    cand = _run(task_ids=("t1", "t3"))
    report = evaluate_gate(base, cand, Tolerances())
    assert any("t2" in w for w in report.warnings)
    assert any("t3" in w for w in report.warnings)


def test_schema_mismatch_warns():
    base = _run(schema_version="2026-01-01.0")
    cand = _run(schema_version="2026-05-29.1")
    report = evaluate_gate(base, cand, Tolerances())
    assert any("schema version" in w for w in report.warnings)


def test_zero_baseline_cost_with_positive_candidate_breaches():
    base = _run(cost=0.0)
    cand = _run(cost=0.01)
    assert evaluate_gate(base, cand, Tolerances()).breached
