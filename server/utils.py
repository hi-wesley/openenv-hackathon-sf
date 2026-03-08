from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict

from dfa_agent_env.config import EVALS_DIR, LOGS_DIR
from dfa_agent_env.models import EmotionScores, ScenarioRecord


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def deterministic_float(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _normalized_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def list_trace_files() -> list[str]:
    candidates = []
    for directory in (LOGS_DIR, EVALS_DIR):
        if directory.exists():
            candidates.extend(str(path) for path in sorted(directory.glob("*.json")))
    return candidates


ASSISTANT_META_MARKERS = (
    "customer expresses",
    "the customer",
    "customer is",
    "user expresses",
    "the user",
    "this conversation",
    "the transcript",
    "conversation shows",
)

CUSTOMER_META_MARKERS = (
    "customer expresses",
    "the customer",
    "assistant says",
    "the assistant",
    "conversation shows",
)

GENERIC_CUSTOMER_LINES = {
    "i still need help with this.",
    "i still need help with this",
    "i still need help.",
    "i still need help",
    "please help.",
    "please help",
}

ASSISTANT_LIKE_CUSTOMER_MARKERS = (
    "i can help",
    "i can review",
    "i will review",
    "let me help",
    "i can arrange",
    "i'll look into",
    "i will check",
    "i'm here to help",
    "your order number",
    "we can expedite",
    "we can process",
    "we can arrange",
    "we can get a replacement",
    "we can replace",
    "we can refund",
    "i don't think we can",
    "i can check",
    "i'll assist you",
    "please provide your",
)


def merge_error_messages(*groups: str | list[str] | tuple[str, ...] | None) -> str | None:
    messages: list[str] = []
    for group in groups:
        if not group:
            continue
        if isinstance(group, str):
            items = [item.strip() for item in group.split(";")]
        else:
            items = [str(item).strip() for item in group]
        for item in items:
            if item and item not in messages:
                messages.append(item)
    return "; ".join(messages) if messages else None


def validate_assistant_message(message: str) -> list[str]:
    normalized = _normalized_line(message)
    if not normalized:
        return ["assistant message must be non-empty"]
    errors: list[str] = []
    if normalized.startswith("customer ") or normalized.startswith("the customer "):
        errors.append("assistant message is narration instead of a direct reply")
    if any(marker in normalized for marker in ASSISTANT_META_MARKERS):
        errors.append("assistant message contains transcript or emotion narration")
    if normalized.startswith("user "):
        errors.append("assistant message refers to the user instead of speaking to them")
    return errors


def validate_customer_message(message: str) -> list[str]:
    normalized = _normalized_line(message)
    if not normalized:
        return ["customer message must be non-empty"]
    errors: list[str] = []
    if normalized in GENERIC_CUSTOMER_LINES:
        errors.append("customer message is generic filler")
    if normalized.startswith("customer ") or any(marker in normalized for marker in CUSTOMER_META_MARKERS):
        errors.append("customer message is narration instead of a first-person reply")
    if any(marker in normalized for marker in ASSISTANT_LIKE_CUSTOMER_MARKERS):
        errors.append("customer message sounds like the support agent")
    if normalized.startswith("i'm really sorry") and any(token in normalized for token in ("replacement", "refund", "order")):
        errors.append("customer message opens like an agent apology instead of a customer complaint")
    if len(normalized.split()) < 5:
        errors.append("customer message is too short to be a realistic reply")
    if not any(token in normalized.split() for token in ("i", "i'm", "im", "my", "me")):
        errors.append("customer message should be written in first person")
    return errors


def coerce_bool(value: Any, *, default: bool) -> bool:
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


def normalize_simulator_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["user_message"] = str(normalized.get("user_message", "")).strip()
    normalized["continue_episode"] = coerce_bool(normalized.get("continue_episode"), default=True)
    normalized["objective_achieved"] = coerce_bool(normalized.get("objective_achieved"), default=False)
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
    validation_errors = validate_customer_message(normalized["user_message"])
    if validation_errors:
        raise ValueError(merge_error_messages(validation_errors) or "Invalid customer message.")
    return normalized


EMOTION_KEYWORDS = {
    "happiness": {
        "glad",
        "great",
        "happy",
        "relieved",
        "perfect",
        "resolved",
        "awesome",
        "excellent",
        "works",
        "wonderful",
    },
    "anger": {
        "angry",
        "furious",
        "ridiculous",
        "unacceptable",
        "outrageous",
        "mad",
        "never",
        "worst",
        "horrible",
    },
    "annoyance": {
        "annoyed",
        "frustrating",
        "frustrated",
        "again",
        "still",
        "inconvenient",
        "waiting",
        "issue",
        "problem",
    },
    "gratitude": {
        "thanks",
        "thank",
        "appreciate",
        "grateful",
        "helpful",
        "appreciated",
    },
}


def score_customer_emotions(text: str) -> EmotionScores:
    lowered = text.lower()
    tokens = re.findall(r"[a-z']+", lowered)
    token_set = set(tokens)
    exclamations = text.count("!")
    question_marks = text.count("?")

    def _score(label: str) -> float:
        hits = sum(1 for keyword in EMOTION_KEYWORDS[label] if keyword in token_set or keyword in lowered)
        base = hits * 0.22
        if label == "anger":
            base += 0.08 * exclamations
            if "not acceptable" in lowered or "this is ridiculous" in lowered:
                base += 0.25
        elif label == "annoyance":
            base += 0.05 * question_marks
            if "still" in token_set and "waiting" in token_set:
                base += 0.18
        elif label == "gratitude" and "thank you" in lowered:
            base += 0.3
        elif label == "happiness" and ("works now" in lowered or "that helps" in lowered):
            base += 0.18
        return clamp(base)

    scores = EmotionScores(
        happiness=_score("happiness"),
        anger=_score("anger"),
        annoyance=_score("annoyance"),
        gratitude=_score("gratitude"),
    )
    if scores.gratitude > 0.0 and scores.anger > 0.0:
        scores.gratitude = clamp(scores.gratitude - 0.12)
    if scores.happiness > 0.0 and scores.annoyance > 0.0:
        scores.happiness = clamp(scores.happiness - 0.08)
    return scores.clipped()


def assistant_message_features(message: str, scenario: ScenarioRecord) -> Dict[str, float]:
    lowered = message.lower()
    empathy = sum(token in lowered for token in ("sorry", "understand", "frustrating", "appreciate", "help"))
    action = sum(token in lowered for token in ("refund", "replace", "replacement", "credit", "cancel", "reverse", "escalate", "manager", "ship", "send"))
    policy = sum(token in lowered for token in ("policy", "cannot", "can't", "unable", "terms"))
    ownership = sum(token in lowered for token in ("i will", "i can", "let me", "i'll", "i am going to"))

    family_bonus = 0.0
    if scenario.family == "late_delivery_refund" and any(token in lowered for token in ("refund", "credit", "reship")):
        family_bonus += 0.3
    if scenario.family == "damaged_item_replacement" and any(token in lowered for token in ("replace", "replacement", "ship a new")):
        family_bonus += 0.3
    if scenario.family == "surprise_billing_cancellation" and any(token in lowered for token in ("refund", "reverse", "charge", "cancel")):
        family_bonus += 0.3

    helpfulness = clamp(0.15 * empathy + 0.12 * action + 0.12 * ownership + family_bonus - 0.18 * policy, -1.0, 1.0)
    return {
        "empathy": float(empathy),
        "action": float(action),
        "policy": float(policy),
        "ownership": float(ownership),
        "helpfulness": helpfulness,
    }


def message_quality_score(message: str, scenario: ScenarioRecord) -> float:
    features = assistant_message_features(message, scenario)
    length_penalty = 0.0
    if len(message.strip()) < 30:
        length_penalty -= 0.08
    if len(message.strip()) > 600:
        length_penalty -= 0.05
    return clamp(0.5 + 0.45 * features["helpfulness"] + length_penalty, 0.0, 1.0)
