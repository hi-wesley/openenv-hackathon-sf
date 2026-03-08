from __future__ import annotations

import argparse
import logging

from dfa_agent_env.compat import HAVE_OPENENV, openenv_create_app
from dfa_agent_env.config import get_config
from dfa_agent_env.models import AssistantAction, DFAObservation
from dfa_agent_env.server.environment import DFAAgentEnvironment
from dfa_agent_env.server.web_ui import build_custom_tab

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_application():
    if not HAVE_OPENENV or openenv_create_app is None:
        raise RuntimeError("openenv-core must be installed to run the DFA Agent server.")
    config = get_config()
    return openenv_create_app(
        env_class=DFAAgentEnvironment,
        action_cls=AssistantAction,
        observation_cls=DFAObservation,
        title=config.app_name,
        description="DFA Agent: multi-turn adaptive assistant benchmark.",
        gradio_builder=build_custom_tab,
    )


app = create_application() if HAVE_OPENENV else None


def main() -> None:  # pragma: no cover - exercised in runtime
    parser = argparse.ArgumentParser(description="Run the DFA Agent OpenEnv server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    import uvicorn

    LOGGER.info("Starting DFA Agent server on %s:%s", args.host, args.port)
    uvicorn.run("dfa_agent_env.server.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":  # pragma: no cover - exercised in runtime
    main()

