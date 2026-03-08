from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable, List

from .config import get_config
from .models import (
    AssistantAction,
    ConversationMessage,
    DFAObservation,
    EpisodeTrace,
    EvalSummaryRow,
    ParseOutcome,
)
from .prompts import COMPACT_TRAIN_TEMPLATE, RICH_DEMO_TEMPLATE


def extract_first_json_object(text: str) -> str | None:
    depth = 0
    start = None
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if char == "\\" and in_string:
            escape = not escape
            continue
        if char == '"' and not escape:
            in_string = not in_string
        escape = False
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : index + 1]
    return None


def normalize_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {"message": str(payload.get("message", "")).strip()}


def parse_action_response(text: str, allow_message_only: bool = False) -> ParseOutcome:
    cfg = get_config()
    raw = text.strip()
    json_blob = extract_first_json_object(raw)
    if json_blob is None:
        if allow_message_only and raw:
            action = AssistantAction.from_message_only(raw)
            errors = action.validate_action(cfg.message_char_budget)
            return ParseOutcome(
                action=None if errors else action,
                parse_error="; ".join(errors) if errors else None,
                raw_text=text,
                used_message_fallback=True,
            )
        return ParseOutcome(action=None, parse_error="No JSON object found in model output.", raw_text=text)
    try:
        payload = json.loads(json_blob)
    except json.JSONDecodeError as exc:
        return ParseOutcome(action=None, parse_error=f"Malformed JSON: {exc}", raw_text=text)
    try:
        action = AssistantAction(**normalize_action_payload(payload))
    except Exception as exc:
        return ParseOutcome(action=None, parse_error=f"Action validation failed: {exc}", raw_text=text)
    errors = action.validate_action(cfg.message_char_budget)
    return ParseOutcome(
        action=None if errors else action,
        parse_error="; ".join(errors) if errors else None,
        raw_text=text,
        used_message_fallback=False,
    )


def _conversation_to_lines(conversation: List[ConversationMessage]) -> str:
    lines = []
    for message in conversation:
        lines.append(f"{message.role.upper()}: {message.content}")
    return "\n".join(lines)


def build_prompt_text(observation: DFAObservation, template_name: str = "compact_train") -> str:
    instruction = COMPACT_TRAIN_TEMPLATE if template_name == "compact_train" else RICH_DEMO_TEMPLATE
    progress_json = json.dumps(observation.task_progress_visible, ensure_ascii=True)
    emotions_json = json.dumps(observation.customer_emotion_scores.model_dump(), ensure_ascii=True)
    convo = _conversation_to_lines(observation.conversation)
    return (
        f"{instruction}\n\n"
        f"Scenario ID: {observation.scenario_id}\n"
        f"Scenario Family: {observation.family}\n"
        f"Turn: {observation.turn_index}/{observation.max_turns}\n"
        f"Visible context: {observation.visible_context}\n"
        f"Current customer emotion scores: {emotions_json}\n"
        f"Current customer satisfaction score: {observation.customer_satisfaction_score:.3f}\n"
        f"Visible progress: {progress_json}\n"
        f"Conversation so far:\n{convo}\n\n"
        f"Latest customer message:\n{observation.latest_user_message}\n"
    )


def trace_to_json(trace: EpisodeTrace, indent: int = 2) -> str:
    return json.dumps(trace.model_dump(), indent=indent, ensure_ascii=False)


def traces_to_jsonl(traces: Iterable[EpisodeTrace]) -> str:
    return "\n".join(json.dumps(trace.model_dump(), ensure_ascii=False) for trace in traces)


def eval_rows_to_csv(rows: Iterable[EvalSummaryRow]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "run_name",
            "split",
            "scenario_id",
            "family",
            "simulator_backend",
            "policy_name",
            "total_reward",
            "shaped_reward",
            "final_satisfaction_reward",
            "task_completion",
            "parse_validity",
            "turns_used",
            "invalid_action_count",
            "customer_summary",
            "metadata",
        ],
    )
    writer.writeheader()
    for row in rows:
        payload = row.model_dump()
        payload["customer_summary"] = json.dumps(payload["customer_summary"], ensure_ascii=False)
        payload["metadata"] = json.dumps(payload["metadata"], ensure_ascii=False)
        writer.writerow(payload)
    return buffer.getvalue()


def save_text(path: str | Path, content: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination
