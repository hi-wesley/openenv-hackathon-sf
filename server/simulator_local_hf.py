from __future__ import annotations

import json

from dfa_agent_env.config import EnvConfig, get_config
from dfa_agent_env.models import SimulatorInput, SimulatorOutput
from dfa_agent_env.prompts import SIMULATOR_OPENING_PROMPT, SIMULATOR_SYSTEM_PROMPT
from dfa_agent_env.serialization import extract_first_json_object
from dfa_agent_env.server.local_hf import generate_chat_text
from dfa_agent_env.server.simulator_base import BaseUserSimulator
from dfa_agent_env.server.utils import normalize_simulator_payload


class LocalHFUserSimulator(BaseUserSimulator):
    name = "local_hf"

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or get_config()

    def generate_opening_message(self, sim_input: SimulatorInput) -> SimulatorOutput:
        prompt = (
            f"{SIMULATOR_OPENING_PROMPT}\n\n"
            f"Scenario:\n{json.dumps(sim_input.scenario.model_dump(), ensure_ascii=False, indent=2)}\n"
        )
        return self._generate(prompt)

    def generate_reply(self, sim_input: SimulatorInput) -> SimulatorOutput:
        # Build a compact conversation transcript for the small local model.
        # Full model_dump() includes metadata (emotion scores, parse errors, etc.)
        # that overwhelms Qwen 0.5B and causes it to repeat earlier messages.
        compact_convo = "\n".join(
            f"{msg.role.upper()}: {msg.content}" for msg in sim_input.conversation
        )
        scenario = sim_input.scenario
        prompt = (
            "Continue this customer-service conversation as the customer.\n"
            "Write the NEXT customer reply. Do NOT repeat any earlier message.\n\n"
            f"Scenario: {scenario.title}\n"
            f"Issue: {scenario.visible_context}\n\n"
            f"Conversation so far:\n{compact_convo}\n\n"
            f"The last assistant message was:\n{sim_input.latest_assistant_message}\n\n"
            "Now write the next customer reply as strict JSON only."
        )
        return self._generate(prompt)

    def _generate(self, prompt: str) -> SimulatorOutput:
        text = ""
        try:
            text = generate_chat_text(
                system_prompt=SIMULATOR_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=self.config.local_simulator_temperature,
                max_new_tokens=self.config.local_model_max_new_tokens,
                config=self.config,
            )
            json_blob = extract_first_json_object(text)
            if json_blob is None:
                raise ValueError("No JSON object in simulator response.")
            payload = normalize_simulator_payload(json.loads(json_blob))
            return SimulatorOutput(**payload, raw_model_output=text)
        except Exception as exc:
            detail = str(exc)
            if text.strip():
                detail = f"{detail} Raw simulator output: {text.strip()[:1200]}"
            return SimulatorOutput(
                user_message="",
                continue_episode=False,
                backend_error=detail,
                simulator_notes=["local-hf backend failure"],
                proxy_signals={"backend_error": 1.0},
                raw_model_output=text or None,
            )
