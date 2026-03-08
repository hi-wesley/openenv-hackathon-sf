from __future__ import annotations

from dfa_agent_env.config import EnvConfig
from dfa_agent_env.models import AssistantAction, PersonaProfile, RewardComponents, SatisfactionScoreResult, SimulatorOutput
from dfa_agent_env.server.utils import clamp, persona_alignment_score


def compute_reward_components(
    *,
    action: AssistantAction,
    persona: PersonaProfile,
    simulator_output: SimulatorOutput,
    parse_valid: bool,
    parse_error: str | None,
    scorer_result: SatisfactionScoreResult | None,
    config: EnvConfig,
) -> RewardComponents:
    alignment = persona_alignment_score(action.strategy_summary(), persona)
    visible_progress = float(simulator_output.visible_progress_update.get("progress_delta", 0.0))
    task_progress_reward = clamp(visible_progress, -1.0, 1.0)
    format_validity_reward = 1.0 if parse_valid else config.invalid_action_penalty
    instruction_following_reward = alignment - (0.15 if parse_error else 0.0)
    satisfaction_score_reward = 0.0
    notes = []
    if scorer_result and scorer_result.available and scorer_result.score is not None:
        satisfaction_score_reward = float(scorer_result.score)
        notes.append("final scorer available")
    if parse_error:
        notes.append(parse_error)
    combined_reward = (
        config.reward_weight_task_progress * task_progress_reward
        + config.reward_weight_format_validity * format_validity_reward
        + config.reward_weight_instruction_following * instruction_following_reward
        + config.reward_weight_satisfaction_score * satisfaction_score_reward
    )
    return RewardComponents(
        task_progress_reward=task_progress_reward,
        format_validity_reward=format_validity_reward,
        instruction_following_reward=instruction_following_reward,
        satisfaction_score_reward=satisfaction_score_reward,
        alignment_proxy=alignment,
        combined_reward=combined_reward,
        notes=notes,
    )

