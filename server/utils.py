from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from dfa_agent_env.config import EVALS_DIR, LOGS_DIR
from dfa_agent_env.models import (
    FormalityPreference,
    HumorTolerance,
    PersonaProfile,
    PersonaLength,
)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def level_to_float(level: str) -> float:
    mapping = {
        "none": 0.0,
        "low": 0.15,
        "short": 0.15,
        "casual": 0.15,
        "medium": 0.5,
        "neutral": 0.5,
        "light": 0.5,
        "high": 0.85,
        "long": 0.85,
        "formal": 0.85,
    }
    return mapping.get(level, 0.5)


def preference_to_level(preference: str) -> str:
    mapping = {
        "short": "low",
        "medium": "medium",
        "long": "high",
        "casual": "low",
        "neutral": "medium",
        "formal": "high",
        "none": "low",
        "light": "medium",
        "high": "high",
    }
    return mapping.get(preference, preference)


def match_score(action_value: str, preferred_value: str) -> float:
    action_f = level_to_float(preference_to_level(action_value))
    target_f = level_to_float(preference_to_level(preferred_value))
    return clamp(1.0 - abs(action_f - target_f) / 0.7, 0.0, 1.0)


def persona_alignment_score(action_summary: Dict[str, Any], persona: PersonaProfile) -> float:
    scores = [
        match_score(str(action_summary.get("verbosity", "medium")), str(persona.preferred_length)),
        match_score(str(action_summary.get("warmth", "medium")), str(persona.warmth_preference)),
        match_score(str(action_summary.get("humor", "low")), str(persona.humor_tolerance)),
        match_score(str(action_summary.get("formality", "medium")), str(persona.formality_preference)),
        match_score(str(action_summary.get("directness", "medium")), str(persona.directness_preference)),
        match_score(str(action_summary.get("initiative", "medium")), str(persona.initiative_preference)),
        match_score(
            str(action_summary.get("explanation_depth", "medium")),
            str(persona.explanation_depth_preference),
        ),
    ]
    return clamp(sum(scores) / len(scores))


def deterministic_float(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def list_trace_files() -> list[str]:
    candidates = []
    for directory in (LOGS_DIR, EVALS_DIR):
        if directory.exists():
            candidates.extend(str(path) for path in sorted(directory.glob("*.json")))
    return candidates

