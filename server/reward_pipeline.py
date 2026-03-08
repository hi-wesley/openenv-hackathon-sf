from __future__ import annotations

from dfa_agent_env.config import EnvConfig
from dfa_agent_env.models import EmotionScores, RewardComponents, SatisfactionScoreResult


def compute_reward_components(
    *,
    previous_emotions: EmotionScores,
    current_emotions: EmotionScores,
    parse_valid: bool,
    parse_error: str | None,
    scorer_result: SatisfactionScoreResult | None,
    config: EnvConfig,
) -> RewardComponents:
    previous_score = previous_emotions.composite()
    current_score = current_emotions.composite()
    score_delta_reward = max(-1.0, min(1.0, current_score - previous_score))
    turn_satisfaction_reward = current_score
    format_validity_reward = 1.0 if parse_valid else config.invalid_action_penalty
    final_satisfaction_reward = 0.0
    notes = []

    if scorer_result and scorer_result.available and scorer_result.score is not None:
        final_satisfaction_reward = float(scorer_result.score)
        notes.append("final scorer available")
    if parse_error:
        notes.append(parse_error)

    combined_reward = (
        config.reward_weight_score_delta * score_delta_reward
        + config.reward_weight_turn_satisfaction * turn_satisfaction_reward
        + config.reward_weight_format_validity * format_validity_reward
        + config.reward_weight_satisfaction_score * final_satisfaction_reward
    )
    return RewardComponents(
        score_delta_reward=score_delta_reward,
        turn_satisfaction_reward=turn_satisfaction_reward,
        format_validity_reward=format_validity_reward,
        final_satisfaction_reward=final_satisfaction_reward,
        current_emotion_scores=current_emotions,
        current_satisfaction_score=current_score,
        combined_reward=combined_reward,
        notes=notes,
    )
