from __future__ import annotations

import importlib
from dataclasses import dataclass

from agent_evals.adapters import AgentAdapter
from agent_evals.tools import ToolRegistry


@dataclass
class AgentPlugin:
    """What a bring-your-own-agent factory returns: an adapter + the tools it runs with.

    The adapter need only satisfy the existing ``AgentAdapter`` protocol —
    ``provider``, ``model``, and ``run_task(task, registry, ctx) -> Trace`` —
    so anything that produces a ``Trace`` can be benchmarked and gated through
    the same scoring path as the built-in providers.
    """

    adapter: AgentAdapter
    registry: ToolRegistry


def load_plugin(spec: str, agent_cfg: object) -> AgentPlugin:
    """Import a ``"module.path:callable"`` factory and call it with the agent config.

    The factory must return an ``AgentPlugin`` (or a duck-typed object exposing
    ``adapter`` and ``registry``). User plugin code is imported into this
    process — only point at code you trust.
    """
    if ":" not in spec:
        raise ValueError(
            f"Invalid plugin spec {spec!r}; expected 'module.path:callable' (e.g. 'myapp.evals:build')."
        )
    module_path, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(f"Could not import plugin module {module_path!r}: {exc}") from exc
    try:
        factory = getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"Plugin module {module_path!r} has no attribute {attr!r}") from exc

    result = factory(agent_cfg)
    adapter = getattr(result, "adapter", None)
    registry = getattr(result, "registry", None)
    if adapter is None or registry is None:
        raise TypeError(
            f"Plugin {spec!r} must return an AgentPlugin (or object with .adapter and .registry); "
            f"got {type(result).__name__}."
        )
    if not hasattr(adapter, "run_task") or not hasattr(adapter, "provider") or not hasattr(adapter, "model"):
        raise TypeError(
            f"Plugin {spec!r} adapter must expose provider, model, and run_task() "
            "(the AgentAdapter protocol)."
        )
    if isinstance(result, AgentPlugin):
        return result
    return AgentPlugin(adapter=adapter, registry=registry)
