from __future__ import annotations

import json
from typing import Any, Callable, Dict

from dfa_agent_env.client import DFAAgentEnv
from dfa_agent_env.models import AssistantAction, EpisodeTrace
from dfa_agent_env.serialization import parse_action_response
from dfa_agent_env.server.environment import DFAAgentEnvironment
from dfa_agent_env.training.prompting import extract_scenario_id


def _to_step_result(result):
    if hasattr(result, "observation"):
        return result
    return type("LocalStepResult", (), {"observation": result, "reward": result.reward, "done": result.done})()


def make_env_adapter(env_url: str | None = None):
    if env_url:
        return DFAAgentEnv(base_url=env_url).sync()
    return DFAAgentEnv(environment=DFAAgentEnvironment()).sync()


def run_text_rollout(
    *,
    env,
    generate_text: Callable[[str, int], str],
    reset_kwargs: Dict[str, Any],
    allow_message_only: bool = True,
) -> dict[str, Any]:
    result = _to_step_result(env.reset(**reset_kwargs))
    prompt_texts: list[str] = []
    completion_texts: list[str] = []
    parse_valid = []
    while not result.done:
        observation = result.observation
        prompt_text = observation.prompt_text
        prompt_texts.append(prompt_text)
        completion_text = generate_text(prompt_text, observation.turn_index)
        completion_texts.append(completion_text)
        parsed = parse_action_response(completion_text, allow_message_only=allow_message_only)
        action = parsed.action or AssistantAction.default("I can help with that. Let me refine it.")
        result = _to_step_result(env.step(action))
        parse_valid.append(float(parsed.parse_error is None))
    state = env.state()
    trace = EpisodeTrace(**state.final_summary["trace"])
    metrics = state.final_summary
    return {
        "trace": trace,
        "env_reward": float(metrics.get("total_reward", 0.0)),
        "task_completion": float(metrics.get("task_completion_flag", 0.0)),
        "parse_valid": sum(parse_valid) / len(parse_valid) if parse_valid else 1.0,
        "prompt_texts": prompt_texts,
        "completion_texts": completion_texts,
        "metrics": metrics,
    }


def run_grpo_rollout_episode(
    *,
    trainer,
    tokenizer,
    prompt_text: str,
    env,
    max_turns: int,
    allow_message_only: bool = False,
) -> dict[str, Any]:
    from trl.experimental.openenv import generate_rollout_completions

    scenario_id = extract_scenario_id(prompt_text)
    reset_kwargs = {
        "split": "train",
        "scenario_id": scenario_id,
        "max_turns": max_turns,
        "seed": 7,
        "mode": "train",
        "simulator_backend": "mock",
    }
    result = _to_step_result(env.reset(**reset_kwargs))
    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    parse_valid = []
    while not result.done and result.observation.turn_index < max_turns:
        current_prompt = result.observation.prompt_text or prompt_text
        rollout_output = generate_rollout_completions(trainer, [current_prompt])[0]
        prompt_ids.extend(rollout_output["prompt_ids"])
        completion_ids.extend(rollout_output["completion_ids"])
        logprobs.extend(rollout_output["logprobs"])
        completion_text = rollout_output.get("text") or tokenizer.decode(
            rollout_output["completion_ids"],
            skip_special_tokens=True,
        )
        parsed = parse_action_response(completion_text, allow_message_only=allow_message_only)
        action = parsed.action or AssistantAction.default("I can help with that. Let me restate it more clearly.")
        result = _to_step_result(env.step(action))
        parse_valid.append(float(parsed.parse_error is None))
    state = env.state()
    trace = EpisodeTrace(**state.final_summary["trace"])
    metrics = state.final_summary
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_reward": float(metrics.get("total_reward", 0.0)),
        "task_completion": float(metrics.get("task_completion_flag", 0.0)),
        "parse_valid": sum(parse_valid) / len(parse_valid) if parse_valid else 1.0,
        "turns_used": float(metrics.get("turns_used", 0)),
        "trace_json": json.dumps(trace.model_dump(), ensure_ascii=False),
    }

