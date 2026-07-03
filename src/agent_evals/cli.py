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

lfd_app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Loss Function Development: run the eval suite as a blinded optimization "
        "target with budgets, instruments, and an iteration log."
    ),
)
app.add_typer(lfd_app, name="lfd")


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


def _load_lfd(config: Path):
    from agent_evals.lfd import load_state
    from agent_evals.project_config import load_project_config

    try:
        cfg = load_project_config(config)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    return cfg, load_state(cfg.lfd.state)


def _print_instruments(state, lfd_cfg, now) -> None:
    """The instruments panel: every budget next to its live reading."""
    elapsed = state.elapsed_h(now)
    table = Table(title="Instruments")
    table.add_column("Instrument")
    table.add_column("Used", justify="right")
    table.add_column("Budget", justify="right")
    table.add_row(
        "wall-clock",
        "-" if elapsed is None else f"{elapsed:.2f}h",
        "-" if lfd_cfg.wallclock_budget_h is None else f"{lfd_cfg.wallclock_budget_h:g}h",
    )
    table.add_row(
        "spend",
        f"${state.spent_usd:.4f}",
        "-" if lfd_cfg.cost_budget_usd is None else f"${lfd_cfg.cost_budget_usd:g}",
    )
    table.add_row(
        "cycles",
        str(len(state.cycles)),
        "-" if lfd_cfg.max_cycles is None else str(lfd_cfg.max_cycles),
    )
    console.print(table)


def _print_score_panel(state, lfd_cfg, *, show_reasons: bool, misses) -> None:
    latest = state.cycles[-1]
    prev = state.cycles[-2].overall if len(state.cycles) > 1 else None
    delta = "" if prev is None else f" ({latest.overall - prev:+.3f} vs prev)"
    best = state.best_overall
    console.print(
        f"cycle {latest.n}: overall={latest.overall:.3f}{delta} | "
        f"target={lfd_cfg.target_overall:g} | best={best:.3f} | "
        f"passed {latest.passed}/{latest.task_count} | "
        f"{latest.duration_s:.1f}s, {latest.tokens} tokens"
        + ("" if latest.cost_usd is None else f", ${latest.cost_usd:.4f}")
    )
    if latest.by_dimension:
        dims = " ".join(f"{name}={score:.3f}" for name, score in sorted(latest.by_dimension.items()))
        console.print(f"dimensions: {dims}")

    if misses:
        table = Table(title=f"Misses ({len(misses)}) — blinded: scores only, no answer key")
        table.add_column("Task")
        table.add_column("Category")
        table.add_column("Overall", justify="right")
        table.add_column("Weak dimensions")
        if show_reasons:
            table.add_column("Judge reasons")
        for miss in misses:
            weak = ", ".join(f"{name}={score:.2f}" for name, score in miss.weak_dimensions.items())
            row = [
                miss.task_id + (" [red](failed)[/red]" if miss.failed else ""),
                miss.category,
                f"{miss.overall:.3f}",
                weak,
            ]
            if show_reasons:
                row.append("; ".join(miss.reasons.values()))
            table.add_row(*row)
        console.print(table)
        if show_reasons:
            console.print(
                "[yellow]Reasons shown: judge rationales can leak eval-shaped hints. "
                "Fixing them one keyword at a time is memorization, not improvement.[/yellow]"
            )


@lfd_app.command("init")
def lfd_init(
    config: Path = typer.Option(Path("agentevals.toml"), help="Path to agentevals.toml."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing goal file."),
) -> None:
    """Write the /goal prompt (GOAL.md) rendered from the [lfd] loss function."""
    from agent_evals.lfd import render_goal

    cfg, _ = _load_lfd(config)
    goal = cfg.lfd.goal
    if goal.exists() and not force:
        console.print(f"[red]{goal} already exists. Use --force to overwrite.[/red]")
        raise typer.Exit(2)
    goal.parent.mkdir(parents=True, exist_ok=True)
    goal.write_text(render_goal(cfg.lfd), encoding="utf-8")
    console.print(f"Wrote {goal}")
    if cfg.lfd.wallclock_budget_h is None or cfg.lfd.cost_budget_usd is None:
        console.print(
            "[yellow]No wall-clock and/or dollar budget set in [lfd] — an unbounded "
            "optimizer will grind forever on marginal gains. Set both before looping.[/yellow]"
        )


@lfd_app.command("score")
def lfd_score(
    hypothesis: str = typer.Option(
        ..., "--hypothesis", "-H", help="What this cycle changed and why it should move the metric."
    ),
    config: Path = typer.Option(Path("agentevals.toml"), help="Path to agentevals.toml."),
    record: str | None = typer.Option(None, help="Override config record mode (live or replay)."),
    out: Path = typer.Option(Path("results/"), help="Where to archive the cycle's run."),
    show_reasons: bool = typer.Option(
        False, "--show-reasons", help="Include judge rationales per miss (may leak eval-shaped hints)."
    ),
) -> None:
    """Run one optimization cycle: execute the agent, score blind, log the cycle.

    Exit codes: 0 target met, 1 below target, 2 setup error, 3 budget exhausted.
    """
    import time
    from datetime import UTC, datetime

    from agent_evals.lfd import evaluate, extract_misses, record_cycle, save_state
    from agent_evals.results_io import write_result
    from agent_evals.runner import execute_run

    cfg, state = _load_lfd(config)
    if record:
        cfg.record = record

    # Budgets are checked before spending anything on a new cycle.
    verdict = evaluate(state, cfg.lfd)
    if verdict.exhausted:
        for reason in verdict.exhausted:
            console.print(f"[red]budget exhausted — {reason}[/red]")
        console.print("[red]Stop optimizing and report where you got to (see `lfd log`).[/red]")
        raise typer.Exit(3)

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

    started = time.monotonic()
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
    duration_s = time.monotonic() - started

    run_file = write_result(candidate, out)
    now = datetime.now(UTC)
    record_cycle(
        state,
        hypothesis=hypothesis,
        run=candidate,
        run_file=run_file,
        duration_s=duration_s,
        now=now,
    )
    save_state(state, cfg.lfd.state)

    misses = extract_misses(candidate)
    _print_score_panel(state, cfg.lfd, show_reasons=show_reasons, misses=misses)
    _print_instruments(state, cfg.lfd, now)

    verdict = evaluate(state, cfg.lfd, now)
    console.print(
        "[dim]Reflect before the next cycle: more general solution, or memorizing the eval? "
        "If memorizing, the next change removes an eval-shaped artifact, not adds one.[/dim]"
    )
    if verdict.target_met:
        console.print(f"[green]TARGET REACHED: overall >= {cfg.lfd.target_overall:g}. Stop optimizing.[/green]")
        return
    if verdict.exhausted:
        for reason in verdict.exhausted:
            console.print(f"[red]budget exhausted — {reason}[/red]")
        console.print("[red]Stop optimizing and report where you got to (see `lfd log`).[/red]")
        raise typer.Exit(3)
    if verdict.stalled:
        console.print(
            f"[yellow]STALLED: {state.stalled_for} cycles without a new best. "
            "The next cycle must be a qualitatively different approach — "
            "not the same knob turned harder.[/yellow]"
        )
    raise typer.Exit(1)


@lfd_app.command("status")
def lfd_status(
    config: Path = typer.Option(Path("agentevals.toml"), help="Path to agentevals.toml."),
) -> None:
    """Where the loop stands: target vs. score, budgets vs. burn, stall state."""
    from datetime import UTC, datetime

    from agent_evals.lfd import evaluate

    cfg, state = _load_lfd(config)
    now = datetime.now(UTC)
    if not state.cycles:
        console.print(
            f"No cycles yet (state: {cfg.lfd.state}). "
            'Run `agent-evals lfd score --hypothesis "..."` to start.'
        )
        return
    latest = state.cycles[-1]
    console.print(
        f"target={cfg.lfd.target_overall:g} | best={state.best_overall:.3f} | "
        f"last={latest.overall:.3f} (cycle {latest.n}, {latest.utc_timestamp})"
    )
    _print_instruments(state, cfg.lfd, now)
    verdict = evaluate(state, cfg.lfd, now)
    if verdict.target_met:
        console.print("[green]Target met.[/green]")
    for reason in verdict.exhausted:
        console.print(f"[red]budget exhausted — {reason}[/red]")
    if verdict.stalled:
        console.print(f"[yellow]Stalled for {state.stalled_for} cycles — force entropy.[/yellow]")


@lfd_app.command("log")
def lfd_log(
    config: Path = typer.Option(Path("agentevals.toml"), help="Path to agentevals.toml."),
    json_out: bool = typer.Option(False, "--json", help="Dump the raw state JSON instead of a table."),
) -> None:
    """The iteration log: every cycle's hypothesis, score, delta, cost, duration."""
    cfg, state = _load_lfd(config)
    if json_out:
        console.print_json(state.model_dump_json())
        return
    table = Table(title=f"Iteration log ({cfg.lfd.state})")
    table.add_column("#", justify="right")
    table.add_column("When")
    table.add_column("Overall", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Hypothesis")
    prev: float | None = None
    for cycle in state.cycles:
        delta = "" if prev is None else f"{cycle.overall - prev:+.3f}"
        prev = cycle.overall
        table.add_row(
            str(cycle.n),
            cycle.utc_timestamp.split(".")[0],
            f"{cycle.overall:.3f}",
            delta,
            "" if cycle.cost_usd is None else f"${cycle.cost_usd:.4f}",
            f"{cycle.duration_s:.1f}s",
            cycle.hypothesis,
        )
    console.print(table)
