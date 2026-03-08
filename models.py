from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from .compat import Action, BaseModel, Field, Observation, State

Level = Literal["low", "medium", "high"]
PersonaLength = Literal["short", "medium", "long"]
HumorTolerance = Literal["none", "light", "high"]
FormalityPreference = Literal["casual", "neutral", "formal"]
AcknowledgementStyle = Literal["none", "brief", "empathetic"]
SplitName = Literal["train", "val", "test"]

LOW_MEDIUM_HIGH: tuple[str, ...] = ("low", "medium", "high")
PREFERRED_LENGTHS: tuple[str, ...] = ("short", "medium", "long")
HUMOR_TOLERANCES: tuple[str, ...] = ("none", "light", "high")
FORMALITY_PREFERENCES: tuple[str, ...] = ("casual", "neutral", "formal")
ACK_STYLES: tuple[str, ...] = ("none", "brief", "empathetic")


def normalize_choice(value: str, choices: tuple[str, ...], default: str | None = None) -> str:
    normalized = (value or "").strip().lower().replace("very_", "").replace("very ", "")
    aliases = {
        "concise": "low",
        "brief": "low",
        "detailed": "high",
        "long": "high",
        "short": "low",
        "direct": "high",
        "indirect": "low",
        "casual": "low",
        "formal": "high",
        "empathetic": "empathetic",
        "empathy": "empathetic",
        "ack": "brief",
        "acknowledge": "brief",
        "lighthearted": "light",
        "none": "none",
        "neutral": "neutral",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in choices:
        return normalized
    if default is not None:
        return default
    raise ValueError(f"Expected one of {choices}, got {value!r}")


class ConversationMessage(BaseModel):
    role: str = Field(description="speaker role")
    content: str = Field(description="message content")
    turn_index: int | None = Field(default=None)
    strategy: Dict[str, Any] | None = Field(default=None)
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
    latent_preference_overrides: Dict[str, Any] = Field(default_factory=dict)
    simulator_instructions: str
    tags: List[str] = Field(default_factory=list)


class PersonaProfile(BaseModel):
    preferred_length: PersonaLength
    warmth_preference: Level
    humor_tolerance: HumorTolerance
    formality_preference: FormalityPreference
    directness_preference: Level
    initiative_preference: Level
    explanation_depth_preference: Level
    expertise_level: Literal["novice", "intermediate", "expert"] = Field(default="intermediate")
    time_pressure: Level = Field(default="medium")
    baseline_patience: float = Field(default=0.6)
    baseline_emotional_state: Literal["calm", "stressed", "upset", "anxious"] = Field(default="calm")

    def reveal_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class HiddenUserState(BaseModel):
    trust: float = Field(default=0.45)
    frustration: float = Field(default=0.25)
    patience_remaining: float = Field(default=0.7)
    emotional_state: str = Field(default="neutral")
    clarity: float = Field(default=0.3)
    goal_progress: float = Field(default=0.0)
    last_visible_progress: float = Field(default=0.0)
    signals: Dict[str, Any] = Field(default_factory=dict)


class AssistantAction(Action):
    verbosity: Level = Field(default="medium")
    warmth: Level = Field(default="medium")
    humor: Level = Field(default="low")
    formality: Level = Field(default="medium")
    directness: Level = Field(default="medium")
    initiative: Level = Field(default="medium")
    explanation_depth: Level = Field(default="medium")
    acknowledgement_style: AcknowledgementStyle = Field(default="brief")
    message: str = Field(min_length=1, description="assistant utterance")

    @classmethod
    def default(cls, message: str = "Let me help with that.") -> "AssistantAction":
        return cls(message=message.strip() or "Let me help with that.")

    @classmethod
    def from_message_only(cls, message: str) -> "AssistantAction":
        return cls.default(message=message)

    def normalized(self) -> "AssistantAction":
        return self.model_copy(
            update={
                "verbosity": normalize_choice(self.verbosity, LOW_MEDIUM_HIGH, "medium"),
                "warmth": normalize_choice(self.warmth, LOW_MEDIUM_HIGH, "medium"),
                "humor": normalize_choice(self.humor, LOW_MEDIUM_HIGH, "low"),
                "formality": normalize_choice(self.formality, LOW_MEDIUM_HIGH, "medium"),
                "directness": normalize_choice(self.directness, LOW_MEDIUM_HIGH, "medium"),
                "initiative": normalize_choice(self.initiative, LOW_MEDIUM_HIGH, "medium"),
                "explanation_depth": normalize_choice(self.explanation_depth, LOW_MEDIUM_HIGH, "medium"),
                "acknowledgement_style": normalize_choice(self.acknowledgement_style, ACK_STYLES, "brief"),
                "message": (self.message or "").strip(),
            },
            deep=True,
        )

    def validate_action(self, message_char_budget: int) -> List[str]:
        errors: List[str] = []
        try:
            normalized = self.normalized()
        except ValueError as exc:
            return [str(exc)]
        if not normalized.message:
            errors.append("message must be non-empty")
        if len(normalized.message) > message_char_budget:
            errors.append(f"message exceeds char budget {message_char_budget}")
        return errors

    def strategy_summary(self) -> Dict[str, Any]:
        payload = self.model_dump()
        payload.pop("message", None)
        payload.pop("metadata", None)
        return payload


class RewardComponents(BaseModel):
    task_progress_reward: float = Field(default=0.0)
    format_validity_reward: float = Field(default=0.0)
    instruction_following_reward: float = Field(default=0.0)
    satisfaction_score_reward: float = Field(default=0.0)
    alignment_proxy: float = Field(default=0.0)
    combined_reward: float = Field(default=0.0)
    notes: List[str] = Field(default_factory=list)


class ParseOutcome(BaseModel):
    action: AssistantAction | None = Field(default=None)
    parse_error: str | None = Field(default=None)
    raw_text: str = Field(default="")
    used_message_fallback: bool = Field(default=False)


class TurnLog(BaseModel):
    turn_index: int
    assistant_action: Dict[str, Any]
    user_message: str
    reward_components: RewardComponents
    parse_valid: bool
    parse_error: str | None = Field(default=None)
    visible_progress: Dict[str, Any] = Field(default_factory=dict)
    hidden_state_snapshot: Dict[str, Any] = Field(default_factory=dict)
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
    persona: PersonaProfile
    conversation: List[ConversationMessage]
    turn_logs: List[TurnLog]
    reward_components: List[RewardComponents]
    final_hidden_state: HiddenUserState
    final_summary: Dict[str, Any]
    mode: str
    simulator_backend: str


class SimulatorInput(BaseModel):
    scenario: ScenarioRecord
    persona: PersonaProfile
    hidden_state: HiddenUserState
    conversation: List[ConversationMessage]
    latest_action: AssistantAction
    latest_assistant_message: str
    turn_index: int
    max_turns: int
    mode: str


class SimulatorOutput(BaseModel):
    user_message: str
    continue_episode: bool
    latent_state_delta: Dict[str, Any] = Field(default_factory=dict)
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
    done_reason: str | None = Field(default=None)
    available_style_axes: List[str] = Field(default_factory=list)
    episode_metrics_visible: Dict[str, Any] = Field(default_factory=dict)
    parse_error: str | None = Field(default=None)
    simulator_backend: str = Field(default="mock")
    prompt_text: str = Field(default="")
    revealable_persona: Dict[str, Any] | None = Field(default=None)


class DFAEnvState(State):
    scenario: ScenarioRecord | None = Field(default=None)
    persona: PersonaProfile | None = Field(default=None)
    hidden_state: HiddenUserState | None = Field(default=None)
    conversation: List[ConversationMessage] = Field(default_factory=list)
    turn_index: int = Field(default=0)
    max_turns: int = Field(default=0)
    task_progress_hidden: Dict[str, Any] = Field(default_factory=dict)
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
    simulator_backend: str = Field(default="mock")
    reveal_persona_after_done: bool = Field(default=False)


class EpisodeTrace(BaseModel):
    episode_id: str | None = Field(default=None)
    scenario: ScenarioRecord
    persona: PersonaProfile
    hidden_state_final: HiddenUserState
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
    persona_summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

