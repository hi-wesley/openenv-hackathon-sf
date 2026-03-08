from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .compat import BaseModel, Field


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else value


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = ROOT_DIR / "docs"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
EVALS_DIR = OUTPUTS_DIR / "evals"


class EnvConfig(BaseModel):
    app_name: str = Field(default="DFA Agent")
    package_name: str = Field(default="dfa_agent_env")
    env_id: str = Field(default="dfa-agent")
    version: str = Field(default="0.1.0")
    openenv_version: str = Field(default="0.2.1")
    default_split: str = Field(default="train")
    default_mode: str = Field(default="demo")
    default_simulator_backend: str = Field(default="mock")
    default_scorer: str = Field(default="noop")
    default_max_turns_demo: int = Field(default=4)
    default_max_turns_train: int = Field(default=4)
    max_turns_limit: int = Field(default=8)
    message_char_budget: int = Field(default=700)
    invalid_action_penalty: float = Field(default=-0.75)
    invalid_action_threshold: int = Field(default=2)
    strict_backend_errors: bool = Field(default=False)
    reveal_persona_after_done_default: bool = Field(default=True)
    reward_weight_task_progress: float = Field(default=0.8)
    reward_weight_format_validity: float = Field(default=0.5)
    reward_weight_instruction_following: float = Field(default=0.5)
    reward_weight_satisfaction_score: float = Field(default=1.0)
    constant_scorer_value: float = Field(default=0.25)
    simulator_base_url: str = Field(default="https://api.openai.com/v1")
    simulator_api_key: str = Field(default="")
    simulator_model: str = Field(default="gpt-4o-mini")
    simulator_temperature: float = Field(default=0.2)
    simulator_timeout_s: float = Field(default=45.0)
    root_dir: str = Field(default=str(ROOT_DIR))
    data_dir: str = Field(default=str(DATA_DIR))
    docs_dir: str = Field(default=str(DOCS_DIR))
    outputs_dir: str = Field(default=str(OUTPUTS_DIR))
    logs_dir: str = Field(default=str(LOGS_DIR))
    evals_dir: str = Field(default=str(EVALS_DIR))


@lru_cache(maxsize=1)
def get_config() -> EnvConfig:
    return EnvConfig(
        app_name=_env_str("DFA_AGENT_APP_NAME", "DFA Agent"),
        package_name=_env_str("DFA_AGENT_PACKAGE_NAME", "dfa_agent_env"),
        env_id=_env_str("DFA_AGENT_ENV_ID", "dfa-agent"),
        version=_env_str("DFA_AGENT_VERSION", "0.1.0"),
        openenv_version=_env_str("DFA_AGENT_OPENENV_VERSION", "0.2.1"),
        default_split=_env_str("DFA_AGENT_DEFAULT_SPLIT", "train"),
        default_mode=_env_str("DFA_AGENT_DEFAULT_MODE", "demo"),
        default_simulator_backend=_env_str("DFA_AGENT_SIMULATOR_BACKEND", "mock"),
        default_scorer=_env_str("DFA_AGENT_SCORER", "noop"),
        default_max_turns_demo=_env_int("DFA_AGENT_MAX_TURNS_DEMO", 4),
        default_max_turns_train=_env_int("DFA_AGENT_MAX_TURNS_TRAIN", 4),
        max_turns_limit=_env_int("DFA_AGENT_MAX_TURNS_LIMIT", 8),
        message_char_budget=_env_int("DFA_AGENT_MESSAGE_CHAR_BUDGET", 700),
        invalid_action_penalty=_env_float("DFA_AGENT_INVALID_ACTION_PENALTY", -0.75),
        invalid_action_threshold=_env_int("DFA_AGENT_INVALID_ACTION_THRESHOLD", 2),
        strict_backend_errors=_env_bool("DFA_AGENT_STRICT_BACKEND_ERRORS", False),
        reveal_persona_after_done_default=_env_bool("DFA_AGENT_REVEAL_PERSONA", True),
        reward_weight_task_progress=_env_float("DFA_AGENT_REWARD_WEIGHT_TASK_PROGRESS", 0.8),
        reward_weight_format_validity=_env_float("DFA_AGENT_REWARD_WEIGHT_FORMAT_VALIDITY", 0.5),
        reward_weight_instruction_following=_env_float(
            "DFA_AGENT_REWARD_WEIGHT_INSTRUCTION_FOLLOWING",
            0.5,
        ),
        reward_weight_satisfaction_score=_env_float("DFA_AGENT_REWARD_WEIGHT_SATISFACTION", 1.0),
        constant_scorer_value=_env_float("DFA_AGENT_CONSTANT_SCORER_VALUE", 0.25),
        simulator_base_url=_env_str("SIMULATOR_BASE_URL", "https://api.openai.com/v1"),
        simulator_api_key=_env_str("SIMULATOR_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        simulator_model=_env_str("SIMULATOR_MODEL", "gpt-4o-mini"),
        simulator_temperature=_env_float("SIMULATOR_TEMPERATURE", 0.2),
        simulator_timeout_s=_env_float("SIMULATOR_TIMEOUT_S", 45.0),
    )


def resolve_output_path(*parts: str) -> Path:
    cfg = get_config()
    path = Path(cfg.outputs_dir).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

