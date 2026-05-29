from agent_evals.models import Category, RunManifest, RunResult, RunSummary, Task
from agent_evals.tasks.loader import load_tasks
from agent_evals.version import SCHEMA_VERSION, __version__


def test_all_tasks_validate():
    tasks = load_tasks()
    assert len(tasks) == 25
    assert {task.category for task in tasks} == set(Category)


def test_run_result_round_trip():
    task = Task(id="x", category="coding", prompt="do x")
    manifest = RunManifest(
        schema_version=SCHEMA_VERSION,
        package_version=__version__,
        provider="openai",
        model="gpt-5.1",
        judge_model="claude-haiku-4-5",
        seed=42,
        tool_layer_version="shared-tools-v1",
        pricing_hash="abc",
        task_hashes={task.id: "hash"},
    )
    run = RunResult(
        schema_version=SCHEMA_VERSION,
        manifest=manifest,
        summary=RunSummary(
            task_count=0,
            overall_mean=0,
            passed=0,
            failure_rate=0,
            total_cost_usd=0,
            median_latency_s=0,
            by_dimension={},
            by_category={},
        ),
        scorecards=[],
    )
    assert RunResult.model_validate_json(run.model_dump_json()).schema_version == SCHEMA_VERSION
