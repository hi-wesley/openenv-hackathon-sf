from __future__ import annotations

from typing import Any, Iterable


def reward_from_env(completions, **kwargs):
    rewards = kwargs.get("env_reward") if kwargs else None
    return [float(item) for item in rewards] if rewards is not None else [0.0] * len(completions)


def metric_from_kwargs(key: str, completions, **kwargs):
    values = kwargs.get(key) if kwargs else None
    return [float(item) for item in values] if values is not None else [0.0] * len(completions)


def summarize_scalar(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0

