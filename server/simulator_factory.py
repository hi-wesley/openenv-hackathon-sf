from __future__ import annotations

from dfa_agent_env.config import EnvConfig, get_config
from dfa_agent_env.server.simulator_base import BaseUserSimulator
from dfa_agent_env.server.simulator_mock import MockUserSimulator
from dfa_agent_env.server.simulator_openai_compatible import OpenAICompatibleUserSimulator


def build_simulator(backend: str | None = None, config: EnvConfig | None = None) -> BaseUserSimulator:
    cfg = config or get_config()
    normalized = (backend or cfg.default_simulator_backend).strip().lower()
    if normalized in {"openai", "real", "openai_compatible"}:
        return OpenAICompatibleUserSimulator(config=cfg)
    return MockUserSimulator()

