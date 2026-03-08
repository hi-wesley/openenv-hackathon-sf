from __future__ import annotations

from typing import Any, Dict, List, Literal

from .compat import Action, BaseModel, Field, Observation, State

SplitName = Literal["train", "val", "test"]


class EmotionScores(BaseModel):
    happiness: float = Field(default=0.0)
    anger: float = Field(default=0.0)
    annoyance: float = Field(default=0.0)
    gratitude: float = Field(default=0.0)

    def clipped(self) -> "EmotionScores":
        def _clip(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        return EmotionScores(
            happiness=_clip(self.happiness),
            anger=_clip(self.anger),
            annoyance=_clip(self.annoyance),
            gratitude=_clip(self.gratitude),
        )

    def composite(self) -> float:
        clipped = self.clipped()
        value = clipped.happiness + clipped.gratitude - clipped.anger - clipped.annoyance
        return max(-2.0, min(2.0, value)) / 2.0

    def to_dict(self) -> Dict[str, float]:
        return self.clipped().model_dump()


class ConversationMessage(BaseModel):
    role: str = Field(description="speaker role")
    content: str = Field(description="message content")
    turn_index: int | None = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScenarioRecord(BaseModel):
    scenario_id: str
    split: SplitName
    family: str
    title: str
    visible_context: str
    initial_user_message: str
    task_success_criteria: List[str]
    allowed_turns: int
    difficulty: str
    simulator_instructions: str
    tags: List[str] = Field(default_factory=list)


class ChatState(BaseModel):
    emotion_scores: EmotionScores = Field(default_factory=EmotionScores)
    satisfaction_score: float = Field(default=0.0)
    objective_achieved: bool = Field(default=False)
    end_requested: bool = Field(default=False)
    signals: Dict[str, Any] = Field(default_factory=dict)


class AssistantAction(Action):
    message: str = Field(min_length=1, description="assistant utterance")

    @classmethod
    def default(cls, message: str = "I can help with that.") -> "AssistantAction":
        return cls(message=(message or "").strip() or "I can help with that.")

    @classmethod
    def from_message_only(cls, message: str) -> "AssistantAction":
        return cls.default(message=message)

    def normalized(self) -> "AssistantAction":
        return self.model_copy(update={"message": (self.message or "").strip()}, deep=True)

    def validate_action(self, message_char_budget: int) -> List[str]:
        normalized = self.normalized()
        errors: List[str] = []
        if not normalized.message:
            errors.append("message must be non-empty")
        if len(normalized.message) > message_char_budget:
            errors.append(f"message exceeds char budget {message_char_budget}")
        return errors

    def summary(self) -> Dict[str, Any]:
        return {
            "message_length": len((self.message or "").strip()),
            "contains_apology": any(
                token in (self.message or "").lower()
                for token in ("sorry", "apologize", "understand", "frustrating")
            ),
        }


class RewardComponents(BaseModel):
    score_delta_reward: float = Field(default=0.0)
    turn_satisfaction_reward: float = Field(default=0.0)
    format_validity_reward: float = Field(default=0.0)
    final_satisfaction_reward: float = Field(default=0.0)
    current_emotion_scores: EmotionScores = Field(default_factory=EmotionScores)
    current_satisfaction_score: float = Field(default=0.0)
    combined_reward: float = Field(default=0.0)
    notes: List[str] = Field(default_factory=list)


class ParseOutcome(BaseModel):
    action: AssistantAction | None = Field(default=None)
    parse_error: str | None = Field(default=None)
    raw_text: str = Field(default="")
    used_message_fallback: bool = Field(default=False)


class TurnLog(BaseModel):
    turn_index: int
    assistant_message: str
    customer_message: str
    customer_emotion_scores: EmotionScores = Field(default_factory=EmotionScores)
    customer_satisfaction_score: float = Field(default=0.0)
    reward_components: RewardComponents
    parse_valid: bool
    parse_error: str | None = Field(default=None)
    visible_progress: Dict[str, Any] = Field(default_factory=dict)
    simulator_notes: List[str] = Field(default_factory=list)
    proxy_signals: Dict[str, Any] = Field(default_factory=dict)
    done: bool = Field(default=False)
    done_reason: str | None = Field(default=None)


class SatisfactionScoreResult(BaseModel):
    score: float | None = Field(default=None)
    available: bool = Field(default=False)
    reason: str | None = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScorerInputs(BaseModel):
    scenario: ScenarioRecord
    conversation: List[ConversationMessage]
    turn_logs: List[TurnLog]
    reward_components: List[RewardComponents]
    final_chat_state: ChatState
    final_summary: Dict[str, Any]
    mode: str
    simulator_backend: str


class SimulatorInput(BaseModel):
    scenario: ScenarioRecord
    conversation: List[ConversationMessage]
    latest_assistant_message: str = Field(default="")
    latest_customer_message: str = Field(default="")
    latest_customer_emotions: EmotionScores = Field(default_factory=EmotionScores)
    turn_index: int
    max_turns: int
    mode: str
    opening_turn: bool = Field(default=False)


class SimulatorOutput(BaseModel):
    user_message: str
    continue_episode: bool
    visible_progress_update: Dict[str, Any] = Field(default_factory=dict)
    simulator_notes: List[str] = Field(default_factory=list)
    proxy_signals: Dict[str, Any] = Field(default_factory=dict)
    objective_achieved: bool = Field(default=False)
    backend_error: str | None = Field(default=None)


class DFAObservation(Observation):
    scenario_id: str
    family: str
    turn_index: int
    max_turns: int
    conversation: List[ConversationMessage]
    latest_user_message: str
    visible_context: str
    assistant_last_action_summary: Dict[str, Any] = Field(default_factory=dict)
    task_progress_visible: Dict[str, Any] = Field(default_factory=dict)
    customer_emotion_scores: EmotionScores = Field(default_factory=EmotionScores)
    customer_satisfaction_score: float = Field(default=0.0)
    done_reason: str | None = Field(default=None)
    episode_metrics_visible: Dict[str, Any] = Field(default_factory=dict)
    parse_error: str | None = Field(default=None)
    simulator_backend: str = Field(default="openai_compatible")
    prompt_text: str = Field(default="")


class DFAEnvState(State):
    scenario: ScenarioRecord | None = Field(default=None)
    chat_state: ChatState | None = Field(default=None)
    conversation: List[ConversationMessage] = Field(default_factory=list)
    turn_index: int = Field(default=0)
    max_turns: int = Field(default=0)
    satisfaction_score: SatisfactionScoreResult | None = Field(default=None)
    per_turn_logs: List[TurnLog] = Field(default_factory=list)
    final_summary: Dict[str, Any] = Field(default_factory=dict)
    reward_components: List[RewardComponents] = Field(default_factory=list)
    simulator_trace: List[Dict[str, Any]] = Field(default_factory=list)
    scorer_inputs: ScorerInputs | None = Field(default=None)
    seed: int | None = Field(default=None)
    mode: str = Field(default="demo")
    invalid_action_count: int = Field(default=0)
    done_reason: str | None = Field(default=None)
    simulator_backend: str = Field(default="openai_compatible")


class EpisodeTrace(BaseModel):
    episode_id: str | None = Field(default=None)
    scenario: ScenarioRecord
    chat_state_final: ChatState
    conversation: List[ConversationMessage]
    turn_logs: List[TurnLog]
    final_summary: Dict[str, Any]
    scorer_result: SatisfactionScoreResult
    reward_components: List[RewardComponents]
    simulator_backend: str
    mode: str
    seed: int | None = Field(default=None)


class EvalSummaryRow(BaseModel):
    run_name: str
    split: str
    scenario_id: str
    family: str
    simulator_backend: str
    policy_name: str
    total_reward: float
    shaped_reward: float
    final_satisfaction_reward: float
    task_completion: float
    parse_validity: float
    turns_used: int
    invalid_action_count: int
    customer_summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
