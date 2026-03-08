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


def _default_simulator_base_url() -> str:
    if os.getenv("SIMULATOR_BASE_URL"):
        return os.getenv("SIMULATOR_BASE_URL", "https://api.openai.com/v1")
    if os.getenv("OPENROUTER_API_KEY"):
        return "https://openrouter.ai/api/v1"
    return "https://api.openai.com/v1"


def _default_simulator_api_key() -> str:
    if os.getenv("SIMULATOR_API_KEY"):
        return os.getenv("SIMULATOR_API_KEY", "")
    if os.getenv("OPENROUTER_API_KEY"):
        return os.getenv("OPENROUTER_API_KEY", "")
    return os.getenv("OPENAI_API_KEY", "")


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = ROOT_DIR / "docs"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
EVALS_DIR = OUTPUTS_DIR / "evals"


def _load_dotenv_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _autoload_local_env() -> None:
    _load_dotenv_file(ROOT_DIR / ".env")


_autoload_local_env()


class EnvConfig(BaseModel):
    app_name: str = Field(default="DFA Agent")
    package_name: str = Field(default="dfa_agent_env")
    env_id: str = Field(default="dfa-agent")
    version: str = Field(default="0.1.0")
    openenv_version: str = Field(default="0.2.1")
    default_split: str = Field(default="train")
    default_mode: str = Field(default="demo")
    default_simulator_backend: str = Field(default="local_hf")
    default_scorer: str = Field(default="emotion_balance")
    default_max_turns_demo: int = Field(default=4)
    default_max_turns_train: int = Field(default=4)
    max_turns_limit: int = Field(default=8)
    message_char_budget: int = Field(default=700)
    invalid_action_penalty: float = Field(default=-0.75)
    invalid_action_threshold: int = Field(default=2)
    strict_backend_errors: bool = Field(default=False)
    reward_weight_turn_satisfaction: float = Field(default=0.75)
    reward_weight_score_delta: float = Field(default=1.0)
    reward_weight_format_validity: float = Field(default=0.5)
    reward_weight_satisfaction_score: float = Field(default=1.0)
    constant_scorer_value: float = Field(default=0.25)
    local_model_id: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    local_model_device: str = Field(default="auto")
    local_model_local_files_only: bool = Field(default=False)
    local_model_max_new_tokens: int = Field(default=256)
    local_assistant_temperature: float = Field(default=0.2)
    local_simulator_temperature: float = Field(default=0.6)
    simulator_base_url: str = Field(default="https://openrouter.ai/api/v1")
    simulator_api_key: str = Field(default="")
    simulator_model: str = Field(default="arcee-ai/trinity-large-preview:free")
    simulator_temperature: float = Field(default=0.2)
    simulator_timeout_s: float = Field(default=45.0)
    openrouter_site_url: str = Field(default="")
    openrouter_app_title: str = Field(default="DFA Agent")
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
        default_simulator_backend=_env_str("DFA_AGENT_SIMULATOR_BACKEND", "local_hf"),
        default_scorer=_env_str("DFA_AGENT_SCORER", "emotion_balance"),
        default_max_turns_demo=_env_int("DFA_AGENT_MAX_TURNS_DEMO", 4),
        default_max_turns_train=_env_int("DFA_AGENT_MAX_TURNS_TRAIN", 4),
        max_turns_limit=_env_int("DFA_AGENT_MAX_TURNS_LIMIT", 8),
        message_char_budget=_env_int("DFA_AGENT_MESSAGE_CHAR_BUDGET", 700),
        invalid_action_penalty=_env_float("DFA_AGENT_INVALID_ACTION_PENALTY", -0.75),
        invalid_action_threshold=_env_int("DFA_AGENT_INVALID_ACTION_THRESHOLD", 2),
        strict_backend_errors=_env_bool("DFA_AGENT_STRICT_BACKEND_ERRORS", False),
        reward_weight_turn_satisfaction=_env_float("DFA_AGENT_REWARD_WEIGHT_TURN_SATISFACTION", 0.75),
        reward_weight_score_delta=_env_float("DFA_AGENT_REWARD_WEIGHT_SCORE_DELTA", 1.0),
        reward_weight_format_validity=_env_float("DFA_AGENT_REWARD_WEIGHT_FORMAT_VALIDITY", 0.5),
        reward_weight_satisfaction_score=_env_float("DFA_AGENT_REWARD_WEIGHT_SATISFACTION", 1.0),
        constant_scorer_value=_env_float("DFA_AGENT_CONSTANT_SCORER_VALUE", 0.25),
        local_model_id=_env_str("DFA_AGENT_LOCAL_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"),
        local_model_device=_env_str("DFA_AGENT_LOCAL_MODEL_DEVICE", "auto"),
        local_model_local_files_only=_env_bool("DFA_AGENT_LOCAL_MODEL_LOCAL_FILES_ONLY", False),
        local_model_max_new_tokens=_env_int("DFA_AGENT_LOCAL_MODEL_MAX_NEW_TOKENS", 256),
        local_assistant_temperature=_env_float("DFA_AGENT_LOCAL_ASSISTANT_TEMPERATURE", 0.2),
        local_simulator_temperature=_env_float("DFA_AGENT_LOCAL_SIMULATOR_TEMPERATURE", 0.6),
        simulator_base_url=_env_str("SIMULATOR_BASE_URL", _default_simulator_base_url()),
        simulator_api_key=_env_str("SIMULATOR_API_KEY", _default_simulator_api_key()),
        simulator_model=_env_str("SIMULATOR_MODEL", "arcee-ai/trinity-large-preview:free"),
        simulator_temperature=_env_float("SIMULATOR_TEMPERATURE", 0.2),
        simulator_timeout_s=_env_float("SIMULATOR_TIMEOUT_S", 45.0),
        openrouter_site_url=_env_str("OPENROUTER_SITE_URL", ""),
        openrouter_app_title=_env_str("OPENROUTER_APP_TITLE", "DFA Agent"),
    )


def resolve_output_path(*parts: str) -> Path:
    cfg = get_config()
    path = Path(cfg.outputs_dir).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
