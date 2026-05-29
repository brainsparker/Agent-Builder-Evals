from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from agent_evals.models import Usage


@dataclass(frozen=True)
class Price:
    input_per_1m: float
    output_per_1m: float
    cache_read_per_1m: float | None = None
    cache_write_per_1m: float | None = None


PRICING: dict[str, Price] = {
    "claude-opus-4-8": Price(5.0, 25.0, 0.5, 6.25),
    "claude-sonnet-4-6": Price(3.0, 15.0, 0.3, 3.75),
    "claude-haiku-4-5": Price(1.0, 5.0, 0.1, 1.25),
    "gpt-5.1": Price(1.25, 10.0),
    "gpt-5-mini": Price(0.25, 2.0),
    "gpt-5-nano": Price(0.05, 0.4),
    "gpt-4.1": Price(2.0, 8.0),
    "gpt-4.1-mini": Price(0.4, 1.6),
}


def pricing_hash() -> str:
    payload = {
        key: price.__dict__
        for key, price in sorted(PRICING.items())
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def cost_for(model: str, usage: Usage) -> tuple[float | None, str | None]:
    price = PRICING.get(model)
    if price is None:
        return None, f"No pricing entry for model {model}"
    input_cost = usage.input_tokens * price.input_per_1m / 1_000_000
    output_cost = usage.output_tokens * price.output_per_1m / 1_000_000
    cache_read_price = price.cache_read_per_1m if price.cache_read_per_1m is not None else price.input_per_1m
    cache_write_price = price.cache_write_per_1m if price.cache_write_per_1m is not None else price.input_per_1m
    cache_read_cost = usage.cache_read_input_tokens * cache_read_price / 1_000_000
    cache_write_cost = usage.cache_creation_input_tokens * cache_write_price / 1_000_000
    return input_cost + output_cost + cache_read_cost + cache_write_cost, None
