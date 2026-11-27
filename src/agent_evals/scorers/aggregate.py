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
from agent_evals.stats import pass_at_k, pass_pow_k, standard_error


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


def cards_by_task(scorecards: list[Scorecard]) -> dict[str, list[Scorecard]]:
    """Group scorecards by task id, keeping first-seen task order."""
    grouped: dict[str, list[Scorecard]] = {}
    for card in scorecards:
        grouped.setdefault(card.task_id, []).append(card)
    return grouped


def task_overall_means(scorecards: list[Scorecard]) -> dict[str, float]:
    """Mean overall score per task across its trials."""
    return {
        task_id: mean(card.overall for card in cards)
        for task_id, cards in cards_by_task(scorecards).items()
    }


def task_dimension_means(scorecards: list[Scorecard], dimension: str) -> dict[str, float]:
    """Mean score per task for one dimension, over the trials that report it."""
    out: dict[str, float] = {}
    for task_id, cards in cards_by_task(scorecards).items():
        scores = [card.dimensions[dimension].score for card in cards if dimension in card.dimensions]
        if scores:
            out[task_id] = mean(scores)
    return out


def task_pass_counts(scorecards: list[Scorecard]) -> dict[str, tuple[int, int]]:
    """``(trials passed, trials run)`` per task."""
    return {
        task_id: (sum(1 for card in cards if card.passed), len(cards))
        for task_id, cards in cards_by_task(scorecards).items()
    }


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
    grouped = cards_by_task(scorecards)
    # Every aggregate score is a mean over tasks of the per-task mean, so a
    # task that happened to get more trials does not weigh more.
    task_means = task_overall_means(scorecards)
    dims = sorted({name for card in scorecards for name in card.dimensions})
    by_dimension = {
        name: mean(task_dimension_means(scorecards, name).values()) for name in dims
    }
    task_category = {task_id: cards[0].category.value for task_id, cards in grouped.items()}
    cats = sorted(set(task_category.values()))
    by_category = {
        cat: mean(score for task_id, score in task_means.items() if task_category[task_id] == cat)
        for cat in cats
    }
    costs = [card.cost_usd for card in scorecards if card.cost_usd is not None]
    counts = task_pass_counts(scorecards)
    trials = max(total for _passed, total in counts.values())
    return RunSummary(
        task_count=len(grouped),
        overall_mean=mean(task_means.values()),
        passed=sum(1 for passed, total in counts.values() if passed == total),
        failure_rate=sum(1 for card in scorecards if card.failed) / len(scorecards),
        total_cost_usd=sum(costs) if len(costs) == len(scorecards) else None,
        median_latency_s=median(card.latency_s for card in scorecards),
        by_dimension=by_dimension,
        by_category=by_category,
        trials=trials,
        pass_rate=mean(passed / total for passed, total in counts.values()),
        pass_at_k=mean(pass_at_k(passed, total, total) for passed, total in counts.values()),
        pass_pow_k=mean(pass_pow_k(passed, total, total) for passed, total in counts.values()),
        overall_se=standard_error(list(task_means.values())),
    )
