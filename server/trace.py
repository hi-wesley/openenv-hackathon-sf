from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dfa_agent_env.models import DFAEnvState, EpisodeTrace, EvalSummaryRow, SatisfactionScoreResult
from dfa_agent_env.serialization import eval_rows_to_csv, save_text, trace_to_json, traces_to_jsonl


def build_episode_trace(state: DFAEnvState) -> EpisodeTrace:
    scorer_result = state.satisfaction_score or SatisfactionScoreResult(
        score=None,
        available=False,
        reason="Trace built before scorer execution.",
    )
    return EpisodeTrace(
        episode_id=state.episode_id,
        scenario=state.scenario,
        persona=state.persona,
        hidden_state_final=state.hidden_state,
        conversation=state.conversation,
        turn_logs=state.per_turn_logs,
        final_summary=state.final_summary,
        scorer_result=scorer_result,
        reward_components=state.reward_components,
        simulator_backend=state.simulator_backend,
        mode=state.mode,
        seed=state.seed,
    )


def export_trace_json(path: str | Path, trace: EpisodeTrace) -> Path:
    return save_text(path, trace_to_json(trace))


def export_traces_jsonl(path: str | Path, traces: Iterable[EpisodeTrace]) -> Path:
    return save_text(path, traces_to_jsonl(traces))


def export_eval_summary_csv(path: str | Path, rows: Iterable[EvalSummaryRow]) -> Path:
    return save_text(path, eval_rows_to_csv(rows))

