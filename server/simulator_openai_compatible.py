from __future__ import annotations

import json
import urllib.request

from dfa_agent_env.config import EnvConfig, get_config
from dfa_agent_env.models import SimulatorInput, SimulatorOutput
from dfa_agent_env.prompts import SIMULATOR_OPENING_PROMPT, SIMULATOR_SYSTEM_PROMPT
from dfa_agent_env.serialization import extract_first_json_object
from dfa_agent_env.server.simulator_base import BaseUserSimulator


class OpenAICompatibleUserSimulator(BaseUserSimulator):
    name = "openai_compatible"

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or get_config()

    def generate_opening_message(self, sim_input: SimulatorInput) -> SimulatorOutput:
        return self._generate(sim_input, opening=True)

    def generate_reply(self, sim_input: SimulatorInput) -> SimulatorOutput:
        return self._generate(sim_input, opening=False)

    def _generate(self, sim_input: SimulatorInput, *, opening: bool) -> SimulatorOutput:
        if not self.config.simulator_api_key:
            return SimulatorOutput(
                user_message="Simulator backend is not configured.",
                continue_episode=False,
                backend_error="Missing SIMULATOR_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY.",
                simulator_notes=["openai-compatible backend unavailable"],
            )
        prompt = self._build_opening_prompt(sim_input) if opening else self._build_reply_prompt(sim_input)
        body = {
            "model": self.config.simulator_model,
            "temperature": self.config.simulator_temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SIMULATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            payload = self._post_json("/chat/completions", body)
            content = payload["choices"][0]["message"]["content"]
            json_blob = extract_first_json_object(content)
            if json_blob is None:
                raise ValueError("No JSON object in simulator response.")
            return SimulatorOutput(**json.loads(json_blob))
        except Exception as exc:
            return SimulatorOutput(
                user_message="The simulator hit a backend error and ended the episode.",
                continue_episode=False,
                backend_error=str(exc),
                simulator_notes=["openai-compatible backend failure"],
                proxy_signals={"backend_error": 1.0},
            )

    def _build_opening_prompt(self, sim_input: SimulatorInput) -> str:
        return (
            f"{SIMULATOR_OPENING_PROMPT}\n\n"
            f"Scenario:\n{json.dumps(sim_input.scenario.model_dump(), ensure_ascii=False, indent=2)}\n"
        )

    def _build_reply_prompt(self, sim_input: SimulatorInput) -> str:
        return (
            "Continue this customer-service conversation as the customer.\n\n"
            f"Scenario:\n{json.dumps(sim_input.scenario.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Latest customer emotions:\n{json.dumps(sim_input.latest_customer_emotions.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Conversation:\n{json.dumps([message.model_dump() for message in sim_input.conversation], ensure_ascii=False, indent=2)}\n\n"
            "Return strict JSON only."
        )

    def _post_json(self, path: str, body: dict) -> dict:
        url = self.config.simulator_base_url.rstrip("/") + path
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.simulator_api_key}",
        }
        if "openrouter.ai" in self.config.simulator_base_url:
            headers["X-Title"] = self.config.openrouter_app_title
            if self.config.openrouter_site_url:
                headers["HTTP-Referer"] = self.config.openrouter_site_url
        request = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.simulator_timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
