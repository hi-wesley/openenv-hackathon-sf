from __future__ import annotations

import hashlib
from typing import Callable, Dict

from .models import AssistantAction, DFAObservation


def _text_flags(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "urgent": any(word in lowered for word in ["urgent", "asap", "quick", "soon", "deadline"]),
        "stressed": any(word in lowered for word in ["stressed", "overwhelmed", "frustrated", "anxious", "panic"]),
        "formal": any(word in lowered for word in ["boss", "client", "professor", "landlord", "refund", "policy"]),
        "wants_brief": any(word in lowered for word in ["brief", "short", "concise"]),
        "wants_detail": any(word in lowered for word in ["detail", "step by step", "explain", "walk me through"]),
    }


def _stable_phrase(seed_text: str, options: list[str]) -> str:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return options[digest[0] % len(options)]


def _base_message(obs: DFAObservation) -> str:
    latest = obs.latest_user_message.strip()
    family = obs.family.replace("_", " ")
    opener = _stable_phrase(
        obs.scenario_id + latest,
        [
            "Here is a draft you can use.",
            "I would handle it like this.",
            "This response should move things forward.",
        ],
    )
    return f"{opener} It addresses the {family} situation while staying grounded in what the user has shared: {latest[:220]}"


def default_policy(observation: DFAObservation) -> AssistantAction:
    flags = _text_flags(observation.latest_user_message)
    verbosity = "low" if flags["wants_brief"] or flags["urgent"] else "high" if flags["wants_detail"] else "medium"
    warmth = "high" if flags["stressed"] else "medium"
    humor = "low" if flags["formal"] or flags["stressed"] else "medium"
    formality = "high" if flags["formal"] else "medium"
    directness = "high" if flags["urgent"] else "medium"
    initiative = "high" if observation.turn_index <= 2 else "medium"
    explanation_depth = "high" if flags["wants_detail"] else "low" if flags["wants_brief"] else "medium"
    acknowledgement_style = "empathetic" if flags["stressed"] else "brief"
    message = _base_message(observation)
    if flags["stressed"]:
        message += " I am keeping the tone calm and practical."
    if flags["urgent"]:
        message += " I am prioritizing the fastest path to a workable next step."
    return AssistantAction(
        verbosity=verbosity,
        warmth=warmth,
        humor=humor,
        formality=formality,
        directness=directness,
        initiative=initiative,
        explanation_depth=explanation_depth,
        acknowledgement_style=acknowledgement_style,
        message=message,
    )


def always_concise_policy(observation: DFAObservation) -> AssistantAction:
    return AssistantAction(
        verbosity="low",
        warmth="medium",
        humor="low",
        formality="medium",
        directness="high",
        initiative="medium",
        explanation_depth="low",
        acknowledgement_style="brief",
        message=f"Short version: {_base_message(observation)}",
    )


def always_warm_detailed_policy(observation: DFAObservation) -> AssistantAction:
    return AssistantAction(
        verbosity="high",
        warmth="high",
        humor="low",
        formality="medium",
        directness="medium",
        initiative="high",
        explanation_depth="high",
        acknowledgement_style="empathetic",
        message=f"I can help with this carefully. {_base_message(observation)} I would also explain the reasoning and give a fuller draft plus next steps.",
    )


def always_formal_direct_policy(observation: DFAObservation) -> AssistantAction:
    return AssistantAction(
        verbosity="medium",
        warmth="low",
        humor="low",
        formality="high",
        directness="high",
        initiative="medium",
        explanation_depth="medium",
        acknowledgement_style="none",
        message=f"Recommended formal response: {_base_message(observation)} The wording should stay clear, specific, and bounded.",
    )


BASELINE_REGISTRY: Dict[str, Callable[[DFAObservation], AssistantAction]] = {
    "default_policy": default_policy,
    "always_concise_policy": always_concise_policy,
    "always_warm_detailed_policy": always_warm_detailed_policy,
    "always_formal_direct_policy": always_formal_direct_policy,
}


def run_baseline(name: str, observation: DFAObservation) -> AssistantAction:
    policy = BASELINE_REGISTRY.get(name, default_policy)
    return policy(observation)

