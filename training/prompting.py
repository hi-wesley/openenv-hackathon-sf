from __future__ import annotations

import re

from dfa_agent_env.models import DFAObservation
from dfa_agent_env.serialization import build_prompt_text


SCENARIO_ID_RE = re.compile(r"^Scenario ID:\s*(?P<scenario_id>\S+)\s*$", re.MULTILINE)


def build_training_prompt(observation: DFAObservation) -> str:
    return build_prompt_text(observation, template_name="compact_train")


def extract_scenario_id(prompt_text: str) -> str | None:
    match = SCENARIO_ID_RE.search(prompt_text)
    return match.group("scenario_id") if match else None

