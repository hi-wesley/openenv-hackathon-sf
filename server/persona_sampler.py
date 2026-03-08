from __future__ import annotations

from dfa_agent_env.models import ChatState, EmotionScores


def initial_chat_state(initial_emotions: EmotionScores | None = None) -> ChatState:
    emotions = (initial_emotions or EmotionScores()).clipped()
    return ChatState(
        emotion_scores=emotions,
        satisfaction_score=emotions.composite(),
        objective_achieved=False,
        end_requested=False,
        signals={},
    )


def update_chat_state(
    current_state: ChatState,
    *,
    customer_emotions: EmotionScores,
    objective_achieved: bool,
    continue_episode: bool,
    extra_signals: dict[str, object] | None = None,
) -> ChatState:
    emotions = customer_emotions.clipped()
    next_state = current_state.model_copy(deep=True)
    next_state.emotion_scores = emotions
    next_state.satisfaction_score = emotions.composite()
    next_state.objective_achieved = bool(objective_achieved)
    next_state.end_requested = not continue_episode
    if extra_signals:
        next_state.signals.update(extra_signals)
    return next_state
