from __future__ import annotations

import json

from dfa_agent_env.config import EnvConfig, get_config
from dfa_agent_env.models import SimulatorInput, SimulatorOutput
from dfa_agent_env.prompts import SIMULATOR_OPENING_PROMPT, SIMULATOR_SYSTEM_PROMPT
from dfa_agent_env.serialization import extract_first_json_object
from dfa_agent_env.server.local_hf import generate_chat_text
from dfa_agent_env.server.simulator_base import BaseUserSimulator


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
        prompt = (
            "Continue this customer-service conversation as the customer.\n\n"
            f"Scenario:\n{json.dumps(sim_input.scenario.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Latest customer emotions:\n{json.dumps(sim_input.latest_customer_emotions.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Conversation:\n{json.dumps([message.model_dump() for message in sim_input.conversation], ensure_ascii=False, indent=2)}\n\n"
            "Return strict JSON only."
        )
        return self._generate(prompt)

    def _generate(self, prompt: str) -> SimulatorOutput:
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
            payload = self._normalize_payload(json.loads(json_blob))
            return SimulatorOutput(**payload)
        except Exception as exc:
            return SimulatorOutput(
                user_message="The local simulator hit an error and ended the episode.",
                continue_episode=False,
                backend_error=str(exc),
                simulator_notes=["local-hf backend failure"],
                proxy_signals={"backend_error": 1.0},
            )

    def _normalize_payload(self, payload: dict) -> dict:
        normalized = dict(payload)
        normalized["user_message"] = str(normalized.get("user_message", "")).strip()
        normalized["continue_episode"] = self._coerce_bool(normalized.get("continue_episode"), default=True)
        normalized["objective_achieved"] = self._coerce_bool(normalized.get("objective_achieved"), default=False)
        visible = normalized.get("visible_progress_update")
        normalized["visible_progress_update"] = visible if isinstance(visible, dict) else {}
        notes = normalized.get("simulator_notes")
        if isinstance(notes, list):
            normalized["simulator_notes"] = [str(item) for item in notes]
        elif isinstance(notes, str) and notes.strip():
            normalized["simulator_notes"] = [notes.strip()]
        else:
            normalized["simulator_notes"] = []
        proxy = normalized.get("proxy_signals")
        normalized["proxy_signals"] = proxy if isinstance(proxy, dict) else {}
        if not normalized["user_message"]:
            normalized["user_message"] = "I still need help with this."
        return normalized

    @staticmethod
    def _coerce_bool(value, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False
            if any(token in lowered for token in ("continue", "assist", "next", "more")):
                return True
            if any(token in lowered for token in ("resolved", "done", "end", "stop")):
                return False
        return default
