from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from agent_evals.models import Category
from agent_evals.results_io import read_result
from agent_evals.runner import replay_result, run_benchmark
from agent_evals.tasks.loader import TASKS_DIR, select_tasks

app = typer.Typer(no_args_is_help=True)
console = Console()


def _model_default(provider: str) -> str:
    return "claude-opus-4-8" if provider == "anthropic" else "gpt-5.1"


@app.command("list-tasks")
def list_tasks(category: Optional[Category] = None) -> None:
    tasks = select_tasks(category=category, all_tasks=category is None)
    table = Table(title="Agent Builder Evals Tasks")
    table.add_column("ID")
    table.add_column("Category")
    table.add_column("Tools")
    for task in tasks:
        table.add_row(task.id, task.category.value, ", ".join(task.allowed_tools))
    console.print(table)


@app.command()
def run(
    provider: str = typer.Option(..., help="anthropic or openai"),
    model: str | None = typer.Option(None),
    tasks: str | None = typer.Option(None, help="Comma-separated task ids."),
    category: Category | None = typer.Option(None),
    all: bool = typer.Option(False, "--all", help="Run all tasks."),
    concurrency: int = typer.Option(1, help="Reserved for future bounded parallelism."),
    judge_model: str = typer.Option("claude-haiku-4-5"),
    record: str = typer.Option("live", help="live or replay"),
    seed: int = typer.Option(42),
    out: Path = typer.Option(Path("results/")),
    tasks_dir: Path | None = typer.Option(None, help="Directory of your own task YAML (defaults to built-ins)."),
) -> None:
    root = tasks_dir or TASKS_DIR
    selected = select_tasks(
        ids=[item.strip() for item in tasks.split(",") if item.strip()] if tasks else None,
        category=category,
        all_tasks=all,
        root=root,
    )
    try:
        path = run_benchmark(
            provider=provider,  # type: ignore[arg-type]
            model=model or _model_default(provider),
            tasks=selected,
            judge_model=judge_model,
            record_mode=record,
            seed=seed,
            out_dir=out,
            concurrency=concurrency,
            tasks_dir=tasks_dir,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("Add missing keys to .env or export them in your shell, then rerun.")
        console.print("Required live search key: YOU_API_KEY")
        raise typer.Exit(1) from exc
    console.print(f"Wrote {path}")


@app.command()
def show(results_file: Path) -> None:
    run = read_result(results_file)
    console.print(
        f"{run.manifest.provider} / {run.manifest.model} | "
        f"overall={run.summary.overall_mean:.3f} passed={run.summary.passed}/{run.summary.task_count}"
    )
    table = Table(title=str(results_file))
    table.add_column("Task")
    table.add_column("Category")
    table.add_column("Overall", justify="right")
    table.add_column("Passed")
    table.add_column("Latency", justify="right")
    table.add_column("Cost", justify="right")
    for card in run.scorecards:
        cost = "" if card.cost_usd is None else f"${card.cost_usd:.4f}"
        table.add_row(
            card.task_id,
            card.category.value,
            f"{card.overall:.3f}",
            "yes" if card.passed else "no",
            f"{card.latency_s:.2f}s",
            cost,
        )
    console.print(table)


@app.command()
def compare(files: list[Path]) -> None:
    runs = [read_result(path) for path in files]
    table = Table(title="Run Comparison")
    table.add_column("Run")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Overall", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("Failure", justify="right")
    table.add_column("Cost", justify="right")
    for path, run in zip(files, runs):
        cost = "" if run.summary.total_cost_usd is None else f"${run.summary.total_cost_usd:.4f}"
        table.add_row(
            path.name,
            run.manifest.provider,
            run.manifest.model,
            f"{run.summary.overall_mean:.3f}",
            f"{run.summary.passed}/{run.summary.task_count}",
            f"{run.summary.failure_rate:.1%}",
            cost,
        )
    console.print(table)


@app.command()
def replay(results_file: Path, out: Path | None = None) -> None:
    path = replay_result(results_file, out)
    console.print(f"Wrote {path}")


def _build_candidate(cfg) -> tuple[object, object, object]:
    """Resolve the agent under test into (adapter, registry, backend).

    A plugin manages its own provider auth; built-in providers require their
    API key (and YOU_API_KEY for live web search). Returns the components the
    runner needs — the candidate is always executed fresh, never replayed.
    """
    from agent_evals.adapters import make_adapter
    from agent_evals.config import load_settings, require_keys
    from agent_evals.runner import default_backend
    from agent_evals.tools import build_default_registry

    if cfg.agent.plugin:
        from agent_evals.plugins import load_plugin

        plugin = load_plugin(cfg.agent.plugin, cfg.agent)
        settings = load_settings()
        if cfg.record == "live" and not settings.you_api_key:
            console.print("[yellow]Warning: record=live but YOU_API_KEY is unset; built-in search tools will fail.[/yellow]")
        return plugin.adapter, plugin.registry, default_backend(cfg.record, settings.you_api_key)

    settings = require_keys(cfg.agent.provider, needs_search=cfg.record == "live")
    api_key = (
        settings.anthropic_api_key if cfg.agent.provider == "anthropic" else settings.openai_api_key
    )
    adapter = make_adapter(
        cfg.agent.provider,
        cfg.agent.model or _model_default(cfg.agent.provider),
        api_key=api_key,
        system_prompt=cfg.agent.system_prompt,
    )
    return adapter, build_default_registry(), default_backend(cfg.record, settings.you_api_key)


@app.command()
def check(
    config: Path = typer.Option(Path("agentevals.toml"), help="Path to agentevals.toml."),
    update_baseline: bool = typer.Option(
        False, "--update-baseline", help="Run the candidate and pin it as the new baseline (no gate)."
    ),
    record: str | None = typer.Option(None, help="Override config record mode (live or replay)."),
    out: Path = typer.Option(Path("results/"), help="Where to archive the candidate run."),
) -> None:
    """Run your agent and fail (nonzero exit) if it regressed against the pinned baseline."""
    from agent_evals.gate import evaluate_gate
    from agent_evals.project_config import load_project_config
    from agent_evals.results_io import write_result, write_result_to
    from agent_evals.runner import execute_run

    try:
        cfg = load_project_config(config)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    if record:
        cfg.record = record

    root = cfg.tasks_dir or TASKS_DIR
    try:
        selected = select_tasks(
            ids=cfg.task_ids,
            category=cfg.category,
            all_tasks=cfg.all_tasks,
            root=root,
        )
        adapter, registry, backend = _build_candidate(cfg)
    except (RuntimeError, ValueError, ImportError, AttributeError, TypeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    candidate = execute_run(
        adapter=adapter,  # type: ignore[arg-type]
        registry=registry,  # type: ignore[arg-type]
        tasks=selected,
        judge_model=cfg.judge_model,
        record_mode=cfg.record,
        seed=cfg.seed,
        backend=backend,
        tasks_dir=cfg.tasks_dir,
    )

    if update_baseline:
        path = write_result_to(candidate, cfg.baseline)
        console.print(
            f"Pinned baseline -> {path} "
            f"(overall={candidate.summary.overall_mean:.3f}, {candidate.summary.task_count} tasks)"
        )
        return

    write_result(candidate, out)
    if not cfg.baseline.exists():
        console.print(
            f"[red]No baseline at {cfg.baseline}. Run `agent-evals check --update-baseline` first.[/red]"
        )
        raise typer.Exit(2)

    baseline = read_result(cfg.baseline)
    report = evaluate_gate(baseline, candidate, cfg.tolerances)

    table = Table(title="Regression gate")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Limit", justify="right")
    table.add_column("Status")
    for row in report.rows:
        status = "[red]BREACH[/red]" if row.breached else "[green]ok[/green]"
        if row.note and not row.breached and row.delta is None:
            status = f"[dim]{row.note}[/dim]"
        delta = "" if row.delta is None else f"{row.delta:+.4f}"
        limit = "" if row.tolerance is None else f"{row.tolerance:+.4f}"
        base = "" if row.baseline is None else f"{row.baseline:.4f}"
        cand = "" if row.candidate is None else f"{row.candidate:.4f}"
        table.add_row(row.name, base, cand, delta, limit, status)
    console.print(table)
    for warning in report.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    if report.breached:
        console.print("[red]Gate FAILED: agent regressed beyond tolerance.[/red]")
        raise typer.Exit(1)
    console.print("[green]Gate passed.[/green]")
