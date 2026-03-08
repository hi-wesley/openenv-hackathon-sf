from __future__ import annotations

import argparse
import logging
import os

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
    os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")
    app = openenv_create_app(
        env=DFAAgentEnvironment,
        action_cls=AssistantAction,
        observation_cls=DFAObservation,
        env_name=config.app_name,
    )
    try:
        import gradio as gr

        dashboard = build_custom_tab(None, metadata={"env_name": config.app_name})
        app = gr.mount_gradio_app(app, dashboard, path="/dashboard")
    except Exception as exc:  # pragma: no cover - runtime-only integration path
        LOGGER.warning("Failed to mount Gradio dashboard at /dashboard: %s", exc)
    app.title = config.app_name
    return app


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
