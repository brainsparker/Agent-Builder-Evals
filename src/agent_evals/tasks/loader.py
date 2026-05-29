from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable

import yaml

from agent_evals.models import Category, Task


TASKS_DIR = Path(__file__).parent / "definitions"


def _task_paths(root: Path = TASKS_DIR) -> list[Path]:
    return sorted(path for path in root.rglob("*.yaml") if path.is_file())


def load_tasks(root: Path = TASKS_DIR) -> list[Task]:
    tasks: list[Task] = []
    for path in _task_paths(root):
        with path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        tasks.append(Task.model_validate(payload))
    return sorted(tasks, key=lambda task: task.id)


def task_hashes(tasks: Iterable[Task], root: Path = TASKS_DIR) -> dict[str, str]:
    hashes: dict[str, str] = {}
    by_id = {task.id: task for task in tasks}
    for path in _task_paths(root):
        with path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        task_id = str(payload.get("id", path.stem))
        if task_id in by_id:
            hashes[task_id] = sha256(path.read_bytes()).hexdigest()[:12]
    return hashes


def select_tasks(
    *,
    ids: list[str] | None = None,
    category: Category | str | None = None,
    all_tasks: bool = False,
    root: Path = TASKS_DIR,
) -> list[Task]:
    tasks = load_tasks(root)
    if all_tasks:
        return tasks
    if ids:
        wanted = set(ids)
        found = [task for task in tasks if task.id in wanted]
        missing = sorted(wanted - {task.id for task in found})
        if missing:
            raise ValueError(f"Unknown task id(s): {', '.join(missing)}")
        return found
    if category:
        cat = Category(category)
        return [task for task in tasks if task.category == cat]
    return tasks
