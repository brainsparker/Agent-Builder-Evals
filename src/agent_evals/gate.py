from __future__ import annotations

from pydantic import BaseModel, Field

from agent_evals.models import RunResult, RunSummary
from agent_evals.scorers.aggregate import (
    task_dimension_means,
    task_overall_means,
    task_pass_counts,
)
from agent_evals.stats import paired_difference

# A breach closer to its limit than this many standard errors draws a warning.
NOISE_WARNING_SIGMA = 2.0


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
    # Score drops (overall and per dimension) must clear their tolerance by
    # this many paired standard errors before they breach. 0 keeps the plain
    # threshold; 2 means "fail only when the regression is clearly not noise".
    noise_sigma: float = Field(default=0.0, ge=0)
    # Allowed drop in pass^k, the share of tasks that passed on every trial.
    # Needs trials > 1 on both runs; leave unset to report without gating.
    consistency_drop: float | None = None


class DeltaRow(BaseModel):
    name: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    tolerance: float | None
    breached: bool
    higher_is_better: bool
    note: str = ""
    # Standard error of the per-task paired difference, when it can be estimated.
    se: float | None = None


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


def _paired_se(baseline: dict[str, float], candidate: dict[str, float]) -> float | None:
    paired = paired_difference(baseline, candidate)
    return None if paired is None else paired[1]


def _score_row(
    *,
    name: str,
    baseline: float,
    candidate: float,
    allowed: float,
    se: float | None,
    tol: Tolerances,
    warnings: list[str],
) -> DeltaRow:
    """Gate one higher-is-better score on its drop, with the noise around it.

    With ``noise_sigma`` set and a standard error available, the drop has to
    clear the tolerance by ``noise_sigma`` standard errors to breach. Without
    it the plain threshold applies, and a breach that sits within two standard
    errors of the limit is flagged as possibly noise instead of being waved
    through.
    """
    delta = candidate - baseline
    margin = tol.noise_sigma * se if se is not None else 0.0
    limit = -allowed - margin
    breached = delta < limit
    note = ""
    if se is not None:
        note = f"±{se:.4f} SE"
        if not breached and delta < -allowed:
            note += f", within {tol.noise_sigma:g} SE of the limit"
        elif breached and tol.noise_sigma == 0 and delta > -allowed - NOISE_WARNING_SIGMA * se:
            warnings.append(
                f"{name} breached by less than {NOISE_WARNING_SIGMA:g} standard errors "
                f"(drop {-delta:.4f}, limit {allowed:.4f}, SE {se:.4f}): this can be "
                "run-to-run noise. Raise `trials`, or set tolerances.noise_sigma to require "
                "a margin."
            )
    elif tol.noise_sigma > 0:
        note = "no SE (needs 2+ shared tasks), plain threshold"
    return DeltaRow(
        name=name,
        baseline=baseline,
        candidate=candidate,
        delta=delta,
        tolerance=limit,
        breached=breached,
        higher_is_better=True,
        note=note,
        se=se,
    )


def evaluate_gate(
    baseline: RunResult,
    candidate: RunResult,
    tol: Tolerances,
) -> GateReport:
    """Diff a candidate run against a pinned baseline and decide pass/fail.

    Reuses the already-computed ``RunSummary`` on each result — no rescoring.
    Score drops also carry the standard error of the per-task paired
    difference, read from the scorecards, so a drop can be judged against the
    noise between two runs of the same tasks.
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

    if b.trials != c.trials:
        warnings.append(
            f"trials differ: baseline={b.trials} candidate={c.trials}; "
            "cost is compared per trial and pass^k is not compared. "
            "Re-pin the baseline with the same trials for a like-for-like gate."
        )

    # Overall — higher is better, gate on the drop.
    overall_se = _paired_se(
        task_overall_means(baseline.scorecards), task_overall_means(candidate.scorecards)
    )
    rows.append(
        _score_row(
            name="overall",
            baseline=b.overall_mean,
            candidate=c.overall_mean,
            allowed=tol.overall_drop,
            se=overall_se,
            tol=tol,
            warnings=warnings,
        )
    )

    # Consistency: pass^k, the share of tasks that passed on every trial.
    if b.trials > 1 and b.trials == c.trials and b.pass_pow_k is not None and c.pass_pow_k is not None:
        delta = c.pass_pow_k - b.pass_pow_k
        allowed = tol.consistency_drop
        rows.append(
            DeltaRow(
                name=f"pass^{c.trials}",
                baseline=b.pass_pow_k,
                candidate=c.pass_pow_k,
                delta=delta,
                tolerance=None if allowed is None else -allowed,
                breached=allowed is not None and delta < -allowed - 1e-12,
                higher_is_better=True,
                note="" if allowed is not None else "not gated (set consistency_drop)",
            )
        )
    elif tol.consistency_drop is not None:
        warnings.append(
            "consistency_drop is set but pass^k was not compared: it needs trials > 1 "
            "and the same trials on the baseline and the candidate"
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
        # Total spend scales with trials, so unequal trials compare per trial.
        per_trial = b.trials != c.trials
        base_cost = b.total_cost_usd / b.trials if per_trial else b.total_cost_usd
        cand_cost = c.total_cost_usd / c.trials if per_trial else c.total_cost_usd
        cost_pct = _pct_increase(base_cost, cand_cost)
        note = f"+{cost_pct:.1%}" if cost_pct != float("inf") else "from $0"
        rows.append(
            DeltaRow(
                name="cost_usd",
                baseline=base_cost,
                candidate=cand_cost,
                delta=cand_cost - base_cost,
                tolerance=tol.cost_increase_pct,
                breached=cost_pct > tol.cost_increase_pct,
                higher_is_better=False,
                note=f"{note} (per trial)" if per_trial else note,
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
        rows.append(
            _score_row(
                name=f"dim:{dim}",
                baseline=b.by_dimension[dim],
                candidate=c.by_dimension[dim],
                allowed=allowed,
                se=_paired_se(
                    task_dimension_means(baseline.scorecards, dim),
                    task_dimension_means(candidate.scorecards, dim),
                ),
                tol=tol,
                warnings=warnings,
            )
        )

    flaky = sorted(
        f"{task_id} {passed}/{total}"
        for task_id, (passed, total) in task_pass_counts(candidate.scorecards).items()
        if 0 < passed < total
    )
    if flaky:
        warnings.append(
            "inconsistent tasks in the candidate (passed some trials, not all): " + ", ".join(flaky)
        )

    breached = any(row.breached for row in rows)
    return GateReport(rows=rows, breached=breached, warnings=warnings)
