"""Small statistics helpers for repeated trials and noise-aware comparisons.

An agent's run is a sequence of sampled model outputs, so one run of one task
reports a property of that run, not of the agent. These helpers turn several
trials per task into rates, and two runs over the same tasks into a paired
difference with a standard error.

References:
- pass@k estimator: Chen et al., "Evaluating Large Language Models Trained on
  Code" (2021), https://arxiv.org/abs/2107.03374
- pass^k: Yao et al., "tau-bench" (2024), https://arxiv.org/abs/2406.12045
- paired differences and standard errors: Miller, "Adding Error Bars to
  Evals" (2024), https://arxiv.org/abs/2411.00640
"""

from __future__ import annotations

from math import comb, sqrt
from statistics import mean, stdev


def pass_at_k(successes: int, trials: int, k: int) -> float:
    """Chance that at least one of ``k`` trials drawn from ``trials`` passed.

    Unbiased counting estimator: ``1 - C(n - c, k) / C(n, k)``.
    """
    _check(successes, trials, k)
    return 1.0 - comb(trials - successes, k) / comb(trials, k)


def pass_pow_k(successes: int, trials: int, k: int) -> float:
    """Chance that all ``k`` trials drawn from ``trials`` passed.

    Unbiased counting estimator: ``C(c, k) / C(n, k)``.
    """
    _check(successes, trials, k)
    return comb(successes, k) / comb(trials, k)


def _check(successes: int, trials: int, k: int) -> None:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError(f"need 0 <= successes <= trials and trials >= 1, got {successes}/{trials}")
    if not 1 <= k <= trials:
        raise ValueError(f"k must be between 1 and trials ({trials}), got {k}")


def standard_error(values: list[float]) -> float | None:
    """Standard error of the mean, or ``None`` with fewer than two values."""
    if len(values) < 2:
        return None
    return stdev(values) / sqrt(len(values))


def paired_difference(
    baseline: dict[str, float], candidate: dict[str, float]
) -> tuple[float, float | None, int] | None:
    """Mean per-task difference (candidate minus baseline) over shared tasks.

    Returns ``(mean_difference, standard_error, task_count)``, or ``None``
    when the two runs share no tasks. Pairing by task removes the variance
    that comes from some tasks simply being harder than others, which is why
    it is the recommended way to compare two runs of the same suite.
    """
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        return None
    diffs = [candidate[task_id] - baseline[task_id] for task_id in shared]
    return mean(diffs), standard_error(diffs), len(shared)
