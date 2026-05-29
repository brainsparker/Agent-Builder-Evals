from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    anthropic_api_key: str | None
    you_api_key: str | None


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        you_api_key=os.getenv("YOU_API_KEY"),
    )


def require_keys(provider: str, *, needs_search: bool = True) -> Settings:
    settings = load_settings()
    missing: list[str] = []
    if provider == "openai" and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if provider == "anthropic" and not settings.anthropic_api_key:
        missing.append("ANTHROPIC_API_KEY")
    if needs_search and not settings.you_api_key:
        missing.append("YOU_API_KEY")
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variable(s): {names}")
    return settings
