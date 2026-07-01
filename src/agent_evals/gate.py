from __future__ import annotations

from pydantic import BaseModel, Field

from agent_evals.models import RunResult, RunSummary


class Tolerances(BaseModel):
    """How much a candidate run is allowed to regress before the gate fails.

    All thresholds are absolute or fractional *worsening* allowances. A delta
    within tolerance passes; anything beyond it breaches the gate.
    """

    overall_drop: float = 0.02
    cost_increase_pct: float = 0.15
    latency_increase_pct: float = 0.20
    fail_rate_increase: float = 0.0
    per_dimension_drop: dict[str, float] = Field(default_factory=dict)
    default_dimension_drop: float | None = None


class DeltaRow(BaseModel):
    name: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    tolerance: float | None
    breached: bool
    higher_is_better: bool
    note: str = ""


class GateReport(BaseModel):
    rows: list[DeltaRow]
    breached: bool
    warnings: list[str] = Field(default_factory=list)


def _pct_increase(baseline: float, candidate: float) -> float:
    """Fractional increase of candidate over baseline.

    Guards a zero baseline: any positive candidate is treated as an infinite
    increase (breaches any finite tolerance), while 0 -> 0 is no change.
    """
    if baseline == 0:
        return 0.0 if candidate <= 0 else float("inf")
    return (candidate - baseline) / baseline


def evaluate_gate(
    baseline: RunResult,
    candidate: RunResult,
    tol: Tolerances,
) -> GateReport:
    """Diff a candidate run against a pinned baseline and decide pass/fail.

    Reuses the already-computed ``RunSummary`` on each result — no rescoring.
    Comparisons are made over the intersection of what both runs measured:
    cost is skipped when either side is ``None`` (mock mode / adapters without
    usage), and per-dimension checks only cover dimensions present in both.
    """
    b: RunSummary = baseline.summary
    c: RunSummary = candidate.summary
    rows: list[DeltaRow] = []
    warnings: list[str] = []

    if baseline.schema_version != candidate.schema_version:
        warnings.append(
            f"schema version differs: baseline={baseline.schema_version} "
            f"candidate={candidate.schema_version}; comparison may be unreliable"
        )

    base_ids = {card.task_id for card in baseline.scorecards}
    cand_ids = {card.task_id for card in candidate.scorecards}
    if base_ids - cand_ids:
        warnings.append(f"tasks in baseline but not candidate: {', '.join(sorted(base_ids - cand_ids))}")
    if cand_ids - base_ids:
        warnings.append(f"new tasks not in baseline (ungated): {', '.join(sorted(cand_ids - base_ids))}")

    # Overall — higher is better, gate on the drop.
    overall_delta = c.overall_mean - b.overall_mean
    rows.append(
        DeltaRow(
            name="overall",
            baseline=b.overall_mean,
            candidate=c.overall_mean,
            delta=overall_delta,
            tolerance=-tol.overall_drop,
            breached=overall_delta < -tol.overall_drop,
            higher_is_better=True,
        )
    )

    # Failure rate — lower is better, gate on the increase.
    fail_delta = c.failure_rate - b.failure_rate
    rows.append(
        DeltaRow(
            name="failure_rate",
            baseline=b.failure_rate,
            candidate=c.failure_rate,
            delta=fail_delta,
            tolerance=tol.fail_rate_increase,
            breached=fail_delta > tol.fail_rate_increase,
            higher_is_better=False,
        )
    )

    # Cost — lower is better; skip entirely if either side is unmeasured.
    if b.total_cost_usd is None or c.total_cost_usd is None:
        rows.append(
            DeltaRow(
                name="cost_usd",
                baseline=b.total_cost_usd,
                candidate=c.total_cost_usd,
                delta=None,
                tolerance=None,
                breached=False,
                higher_is_better=False,
                note="skipped (cost unmeasured)",
            )
        )
    else:
        cost_pct = _pct_increase(b.total_cost_usd, c.total_cost_usd)
        rows.append(
            DeltaRow(
                name="cost_usd",
                baseline=b.total_cost_usd,
                candidate=c.total_cost_usd,
                delta=c.total_cost_usd - b.total_cost_usd,
                tolerance=tol.cost_increase_pct,
                breached=cost_pct > tol.cost_increase_pct,
                higher_is_better=False,
                note=f"+{cost_pct:.1%}" if cost_pct != float("inf") else "from $0",
            )
        )

    # Latency — lower is better.
    lat_pct = _pct_increase(b.median_latency_s, c.median_latency_s)
    rows.append(
        DeltaRow(
            name="median_latency_s",
            baseline=b.median_latency_s,
            candidate=c.median_latency_s,
            delta=c.median_latency_s - b.median_latency_s,
            tolerance=tol.latency_increase_pct,
            breached=lat_pct > tol.latency_increase_pct,
            higher_is_better=False,
            note=f"+{lat_pct:.1%}" if lat_pct != float("inf") else "from 0s",
        )
    )

    # Per-dimension — higher is better; only dimensions present in both sides.
    shared_dims = sorted(set(b.by_dimension) & set(c.by_dimension))
    for dim in shared_dims:
        allowed = tol.per_dimension_drop.get(dim, tol.default_dimension_drop)
        if allowed is None:
            continue
        delta = c.by_dimension[dim] - b.by_dimension[dim]
        rows.append(
            DeltaRow(
                name=f"dim:{dim}",
                baseline=b.by_dimension[dim],
                candidate=c.by_dimension[dim],
                delta=delta,
                tolerance=-allowed,
                breached=delta < -allowed,
                higher_is_better=True,
            )
        )

    breached = any(row.breached for row in rows)
    return GateReport(rows=rows, breached=breached, warnings=warnings)
