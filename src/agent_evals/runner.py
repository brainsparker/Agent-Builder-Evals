from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent_evals.adapters import make_adapter
from agent_evals.config import require_keys
from agent_evals.git import git_revision
from agent_evals.models import ProviderName, RunManifest, RunResult, Task
from agent_evals.pricing import pricing_hash
from agent_evals.results_io import write_result
from agent_evals.scorers.aggregate import score_trace, summarize
from agent_evals.tasks.loader import task_hashes
from agent_evals.tools import build_default_registry
from agent_evals.tools.base import ToolContext
from agent_evals.tools.search_backend import MockBackend, YouBackend
from agent_evals.version import SCHEMA_VERSION, __version__


def run_benchmark(
    *,
    provider: ProviderName,
    model: str,
    tasks: list[Task],
    judge_model: str,
    record_mode: str,
    seed: int,
    out_dir: Path,
    concurrency: int = 1,
) -> Path:
    settings = require_keys(provider, needs_search=record_mode == "live")
    api_key = settings.anthropic_api_key if provider == "anthropic" else settings.openai_api_key
    backend = YouBackend(settings.you_api_key or "") if record_mode == "live" else MockBackend()
    adapter = make_adapter(provider, model, api_key=api_key)
    registry = build_default_registry()
    cassette: dict[str, object] = {}
    def run_one(task: Task):
        ctx = ToolContext(search_backend=backend, record_mode=record_mode, cassette=cassette)
        trace = adapter.run_task(task, registry, ctx)
        return score_trace(task, trace)

    if concurrency <= 1 or len(tasks) <= 1:
        scorecards = [run_one(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            scorecards = list(pool.map(run_one, tasks))
    manifest = RunManifest(
        schema_version=SCHEMA_VERSION,
        package_version=__version__,
        git_revision=git_revision(),
        provider=provider,
        model=model,
        judge_model=judge_model,
        seed=seed,
        tool_layer_version="shared-tools-v1",
        pricing_hash=pricing_hash(),
        task_hashes=task_hashes(tasks),
        record_mode=record_mode,  # type: ignore[arg-type]
    )
    run = RunResult(
        schema_version=SCHEMA_VERSION,
        manifest=manifest,
        summary=summarize(scorecards),
        scorecards=scorecards,
        cassette=cassette,
    )
    return write_result(run, out_dir)


def replay_result(path: Path, out_dir: Path | None = None) -> Path:
    from agent_evals.results_io import read_result
    from agent_evals.tasks.loader import load_tasks

    prior = read_result(path)
    tasks_by_id = {task.id: task for task in load_tasks()}
    scorecards = [
        score_trace(tasks_by_id[card.task_id], card.trace)
        for card in prior.scorecards
        if card.task_id in tasks_by_id
    ]
    manifest = prior.manifest.model_copy(update={"record_mode": "replay"})
    replayed = RunResult(
        schema_version=prior.schema_version,
        manifest=manifest,
        summary=summarize(scorecards),
        scorecards=scorecards,
        cassette=prior.cassette,
    )
    return write_result(replayed, out_dir or path.parent)
