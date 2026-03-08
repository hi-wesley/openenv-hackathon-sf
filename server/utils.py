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


def list_trace_files() -> list[str]:
    candidates = []
    for directory in (LOGS_DIR, EVALS_DIR):
        if directory.exists():
            candidates.extend(str(path) for path in sorted(directory.glob("*.json")))
    return candidates


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
