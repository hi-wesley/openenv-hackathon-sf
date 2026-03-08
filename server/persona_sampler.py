from __future__ import annotations

import random

from dfa_agent_env.models import AssistantAction, HiddenUserState, PersonaProfile, ScenarioRecord
from dfa_agent_env.server.utils import clamp, match_score, persona_alignment_score


def _sample_choice(rng: random.Random, values: list[str], weights: list[float]) -> str:
    total = sum(weights)
    point = rng.random() * total
    running = 0.0
    for value, weight in zip(values, weights):
        running += weight
        if point <= running:
            return value
    return values[-1]


def sample_persona(
    rng: random.Random,
    scenario: ScenarioRecord,
    difficulty: str | None = None,
) -> PersonaProfile:
    preferred_length = _sample_choice(rng, ["short", "medium", "long"], [0.35, 0.4, 0.25])
    warmth_preference = _sample_choice(rng, ["low", "medium", "high"], [0.2, 0.45, 0.35])
    humor_tolerance = _sample_choice(rng, ["none", "light", "high"], [0.45, 0.4, 0.15])
    formality_preference = _sample_choice(rng, ["casual", "neutral", "formal"], [0.25, 0.45, 0.3])
    directness_preference = _sample_choice(rng, ["low", "medium", "high"], [0.2, 0.45, 0.35])
    initiative_preference = _sample_choice(rng, ["low", "medium", "high"], [0.2, 0.45, 0.35])
    explanation_depth_preference = _sample_choice(rng, ["low", "medium", "high"], [0.2, 0.45, 0.35])
    expertise_level = _sample_choice(rng, ["novice", "intermediate", "expert"], [0.45, 0.4, 0.15])
    time_pressure = _sample_choice(rng, ["low", "medium", "high"], [0.2, 0.45, 0.35])
    emotional_state = _sample_choice(rng, ["calm", "stressed", "upset", "anxious"], [0.3, 0.3, 0.15, 0.25])
    baseline_patience = {"easy": 0.78, "medium": 0.65, "hard": 0.52}.get(difficulty or scenario.difficulty, 0.65)
    persona = PersonaProfile(
        preferred_length=preferred_length,
        warmth_preference=warmth_preference,
        humor_tolerance=humor_tolerance,
        formality_preference=formality_preference,
        directness_preference=directness_preference,
        initiative_preference=initiative_preference,
        explanation_depth_preference=explanation_depth_preference,
        expertise_level=expertise_level,
        time_pressure=time_pressure,
        baseline_patience=baseline_patience,
        baseline_emotional_state=emotional_state,
    )
    overrides = scenario.latent_preference_overrides or {}
    return persona.model_copy(update=overrides)


def initial_hidden_state(persona: PersonaProfile, scenario: ScenarioRecord) -> HiddenUserState:
    return HiddenUserState(
        trust=0.38 if scenario.difficulty == "hard" else 0.48,
        frustration=0.42 if persona.baseline_emotional_state in {"stressed", "upset", "anxious"} else 0.22,
        patience_remaining=persona.baseline_patience,
        emotional_state=persona.baseline_emotional_state,
        clarity=0.22 if persona.expertise_level == "novice" else 0.38,
        goal_progress=0.0,
        last_visible_progress=0.0,
        signals={"scenario_family": scenario.family},
    )


def apply_assistant_action(
    hidden_state: HiddenUserState,
    persona: PersonaProfile,
    action: AssistantAction,
) -> HiddenUserState:
    normalized = action.normalized()
    alignment = persona_alignment_score(normalized.strategy_summary(), persona)
    next_state = hidden_state.model_copy(deep=True)
    next_state.trust = clamp(next_state.trust + 0.06 * (alignment - 0.5))
    next_state.frustration = clamp(next_state.frustration + 0.07 * (0.5 - alignment))
    next_state.patience_remaining = clamp(next_state.patience_remaining - 0.02 - 0.03 * match_score(normalized.verbosity, "high"))
    if persona.time_pressure == "high" and normalized.verbosity == "high":
        next_state.frustration = clamp(next_state.frustration + 0.08)
    if persona.warmth_preference == "high" and normalized.warmth == "high":
        next_state.trust = clamp(next_state.trust + 0.04)
    next_state.signals["assistant_alignment"] = alignment
    return next_state


def apply_simulator_delta(hidden_state: HiddenUserState, delta: dict[str, object]) -> HiddenUserState:
    next_state = hidden_state.model_copy(deep=True)
    for key, value in delta.items():
        if not hasattr(next_state, key):
            next_state.signals[key] = value
            continue
        current = getattr(next_state, key)
        if isinstance(current, (int, float)) and isinstance(value, (int, float)):
            setattr(next_state, key, clamp(float(current) + float(value)))
        else:
            setattr(next_state, key, value)
    return next_state

