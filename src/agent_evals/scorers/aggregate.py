from __future__ import annotations

from statistics import mean, median

from agent_evals.models import DimensionScore, RunSummary, Scorecard, Task, Trace
from agent_evals.pricing import cost_for
from agent_evals.scorers.citation_quality import CitationQualityScorer
from agent_evals.scorers.coding import CodingHiddenTestsScorer
from agent_evals.scorers.cost import CostScorer
from agent_evals.scorers.failure import FailureScorer
from agent_evals.scorers.latency import LatencyScorer
from agent_evals.scorers.task_completion import TaskCompletionScorer
from agent_evals.scorers.tool_accuracy import ToolAccuracyScorer


DEFAULT_WEIGHTS = {
    "completion": 0.40,
    "citation": 0.20,
    "tool": 0.15,
    "failure": 0.10,
    "cost": 0.075,
    "latency": 0.075,
    "coding_tests": 0.40,
}


def score_trace(task: Task, trace: Trace) -> Scorecard:
    scorers = [
        TaskCompletionScorer(),
        CitationQualityScorer(),
        ToolAccuracyScorer(),
        FailureScorer(),
        CostScorer(),
        LatencyScorer(),
        CodingHiddenTestsScorer(),
    ]
    configured = {**DEFAULT_WEIGHTS, **task.scoring.weights}
    results = [scorer.score(task, trace) for scorer in scorers]
    applicable = [result for result in results if result.applicable]
    total_weight = sum(configured.get(result.name, 0) for result in applicable) or 1
    dimensions: dict[str, DimensionScore] = {}
    overall = 0.0
    for result in results:
        weight = configured.get(result.name, 0)
        normalized_weight = weight / total_weight if result.applicable else 0
        dimensions[result.name] = DimensionScore(
            score=result.score,
            raw=result.raw,
            weight=normalized_weight,
            reason=result.reason,
        )
        if result.applicable:
            overall += normalized_weight * result.score
    cost, _warning = cost_for(trace.model, trace.usage)
    return Scorecard(
        task_id=task.id,
        category=task.category,
        provider=trace.provider,
        model=trace.model,
        dimensions=dimensions,
        overall=round(overall, 4),
        passed=overall >= task.scoring.pass_threshold,
        cost_usd=cost,
        latency_s=trace.latency_s,
        failed=trace.finished_reason != "completed" or trace.error is not None,
        trace=trace,
    )


def summarize(scorecards: list[Scorecard]) -> RunSummary:
    if not scorecards:
        return RunSummary(
            task_count=0,
            overall_mean=0,
            passed=0,
            failure_rate=0,
            total_cost_usd=0,
            median_latency_s=0,
            by_dimension={},
            by_category={},
        )
    dims = sorted({name for card in scorecards for name in card.dimensions})
    by_dimension = {
        name: mean(card.dimensions[name].score for card in scorecards if name in card.dimensions)
        for name in dims
    }
    cats = sorted({card.category.value for card in scorecards})
    by_category = {
        cat: mean(card.overall for card in scorecards if card.category.value == cat)
        for cat in cats
    }
    costs = [card.cost_usd for card in scorecards if card.cost_usd is not None]
    return RunSummary(
        task_count=len(scorecards),
        overall_mean=mean(card.overall for card in scorecards),
        passed=sum(1 for card in scorecards if card.passed),
        failure_rate=sum(1 for card in scorecards if card.failed) / len(scorecards),
        total_cost_usd=sum(costs) if len(costs) == len(scorecards) else None,
        median_latency_s=median(card.latency_s for card in scorecards),
        by_dimension=by_dimension,
        by_category=by_category,
    )
