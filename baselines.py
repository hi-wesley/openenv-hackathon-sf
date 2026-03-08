from __future__ import annotations

import hashlib
from typing import Callable, Dict

from .models import AssistantAction, DFAObservation


def _stable_phrase(seed_text: str, options: list[str]) -> str:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return options[digest[0] % len(options)]


def _acknowledgement(observation: DFAObservation) -> str:
    if observation.customer_emotion_scores.anger >= 0.5 or observation.customer_emotion_scores.annoyance >= 0.5:
        return _stable_phrase(
            observation.scenario_id + observation.latest_user_message,
            [
                "I understand why this is frustrating.",
                "I’m sorry this has been such a frustrating experience.",
                "I can see why you’re upset about this.",
            ],
        )
    return _stable_phrase(
        observation.scenario_id + observation.latest_user_message,
        [
            "Thanks for explaining what happened.",
            "I appreciate the context.",
            "Thanks for spelling that out clearly.",
        ],
    )


def default_policy(observation: DFAObservation) -> AssistantAction:
    family = observation.family
    latest = observation.latest_user_message
    prefix = _acknowledgement(observation)
    if family == "late_delivery_refund":
        message = (
            f"{prefix} I want to help move this forward. "
            "I would confirm the order details, check the shipping failure, and if the delay has made the order useless for you, I would move toward a refund or make-good right away."
        )
    elif family == "damaged_item_replacement":
        message = (
            f"{prefix} This should not have arrived in that condition. "
            "I would verify the order and start a replacement as quickly as possible so you are not stuck with a damaged item."
        )
    else:
        message = (
            f"{prefix} I want to sort out the unexpected charge. "
            "I would review the cancellation and billing timeline, explain what happened clearly, and if the charge was incorrect, move toward reversing it."
        )
    if latest.endswith("?"):
        message += " I’ll keep this simple and focused on the next step."
    return AssistantAction(message=message)


def empathetic_policy(observation: DFAObservation) -> AssistantAction:
    return AssistantAction(
        message=(
            f"{_acknowledgement(observation)} You should not have to chase this down. "
            "I’ll focus on a clear resolution and keep the next steps simple."
        )
    )


def refund_first_policy(observation: DFAObservation) -> AssistantAction:
    return AssistantAction(
        message=(
            f"{_acknowledgement(observation)} If this situation warrants it, "
            "I would prioritize a refund or billing correction instead of asking you to repeat the whole story."
        )
    )


def concise_policy(observation: DFAObservation) -> AssistantAction:
    return AssistantAction(
        message=(
            f"{_acknowledgement(observation)} I’ll review the case and move to the fastest fix."
        )
    )


BASELINE_REGISTRY: Dict[str, Callable[[DFAObservation], AssistantAction]] = {
    "default_policy": default_policy,
    "empathetic_policy": empathetic_policy,
    "refund_first_policy": refund_first_policy,
    "concise_policy": concise_policy,
}


def run_baseline(name: str, observation: DFAObservation) -> AssistantAction:
    policy = BASELINE_REGISTRY.get(name, default_policy)
    return policy(observation)
