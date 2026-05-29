from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from agent_evals.models import Category
from agent_evals.results_io import read_result
from agent_evals.runner import replay_result, run_benchmark
from agent_evals.tasks.loader import select_tasks

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
) -> None:
    selected = select_tasks(
        ids=[item.strip() for item in tasks.split(",") if item.strip()] if tasks else None,
        category=category,
        all_tasks=all,
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
