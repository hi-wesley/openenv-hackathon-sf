from __future__ import annotations

import json
import urllib.error
import urllib.request

from dfa_agent_env.config import EnvConfig, get_config
from dfa_agent_env.models import SimulatorInput, SimulatorOutput
from dfa_agent_env.prompts import SIMULATOR_SYSTEM_PROMPT
from dfa_agent_env.serialization import extract_first_json_object
from dfa_agent_env.server.simulator_base import BaseUserSimulator


class OpenAICompatibleUserSimulator(BaseUserSimulator):
    name = "openai_compatible"

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or get_config()

    def generate_reply(self, sim_input: SimulatorInput) -> SimulatorOutput:
        if not self.config.simulator_api_key:
            return SimulatorOutput(
                user_message="Simulator backend is not configured.",
                continue_episode=False,
                backend_error="Missing SIMULATOR_API_KEY or OPENAI_API_KEY.",
                simulator_notes=["openai-compatible backend unavailable"],
            )
        body = {
            "model": self.config.simulator_model,
            "temperature": self.config.simulator_temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SIMULATOR_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(sim_input)},
            ],
        }
        try:
            payload = self._post_json("/chat/completions", body)
            content = payload["choices"][0]["message"]["content"]
            json_blob = extract_first_json_object(content)
            if json_blob is None:
                raise ValueError("No JSON object in simulator response.")
            parsed = json.loads(json_blob)
            return SimulatorOutput(**parsed)
        except Exception as exc:
            return SimulatorOutput(
                user_message="The simulator hit a backend error and ended the episode.",
                continue_episode=False,
                backend_error=str(exc),
                simulator_notes=["openai-compatible backend failure"],
                proxy_signals={"backend_error": 1.0},
            )

    def _build_user_prompt(self, sim_input: SimulatorInput) -> str:
        return json.dumps(sim_input.model_dump(), ensure_ascii=False, indent=2)

    def _post_json(self, path: str, body: dict) -> dict:
        url = self.config.simulator_base_url.rstrip("/") + path
        request = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.simulator_api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.simulator_timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

