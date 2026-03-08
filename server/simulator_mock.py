from __future__ import annotations

from typing import Dict

from dfa_agent_env.models import SimulatorInput, SimulatorOutput
from dfa_agent_env.server.simulator_base import BaseUserSimulator
from dfa_agent_env.server.utils import clamp, persona_alignment_score


FAMILY_KEYWORDS: Dict[str, list[str]] = {
    "tough_email_reply": ["draft", "email", "subject", "tone", "reply"],
    "schedule_conflict_planning": ["reschedule", "available", "option", "calendar", "time"],
    "troubleshooting_support": ["check", "restart", "settings", "step", "account"],
    "recommendation_request": ["option", "recommend", "because", "budget", "fit"],
    "negotiation_exception_request": ["request", "refund", "extension", "policy", "exception"],
    "emotionally_charged_message_handling": ["acknowledge", "support", "feel", "next step", "message"],
}


class MockUserSimulator(BaseUserSimulator):
    name = "mock"

    def generate_reply(self, sim_input: SimulatorInput) -> SimulatorOutput:
        action = sim_input.latest_action.normalized()
        persona = sim_input.persona
        hidden = sim_input.hidden_state
        alignment = persona_alignment_score(action.strategy_summary(), persona)
        message_lc = action.message.lower()
        keyword_bonus = 0.0
        for keyword in FAMILY_KEYWORDS.get(sim_input.scenario.family, []):
            if keyword in message_lc:
                keyword_bonus += 0.04
        if persona.time_pressure == "high" and action.verbosity == "high":
            keyword_bonus -= 0.08
        if persona.expertise_level == "novice" and action.explanation_depth == "low":
            keyword_bonus -= 0.1
        progress_delta = clamp(0.08 + 0.35 * alignment + keyword_bonus - 0.1 * hidden.frustration, -0.2, 0.5)
        trust_delta = clamp(0.14 * (alignment - 0.4), -0.2, 0.2)
        frustration_delta = clamp(0.12 * (0.5 - alignment), -0.2, 0.2)
        continue_episode = sim_input.turn_index + 1 < sim_input.max_turns
        objective_achieved = hidden.goal_progress + progress_delta >= 0.78 or hidden.trust + trust_delta >= 0.82
        user_message = self._build_user_message(sim_input, alignment, progress_delta, objective_achieved)
        if objective_achieved:
            continue_episode = False
        if hidden.frustration + max(frustration_delta, 0.0) >= 0.9:
            user_message = "This still is not quite working for me, and I need to stop here for now."
            continue_episode = False
        return SimulatorOutput(
            user_message=user_message,
            continue_episode=continue_episode,
            latent_state_delta={
                "trust": trust_delta,
                "frustration": frustration_delta,
                "goal_progress": progress_delta,
                "last_visible_progress": progress_delta,
                "patience_remaining": -0.05,
                "clarity": 0.08 * alignment,
                "emotional_state": "relieved" if objective_achieved else hidden.emotional_state,
            },
            visible_progress_update={
                "progress_delta": round(progress_delta, 3),
                "goal_progress_hint": round(clamp(hidden.goal_progress + progress_delta), 3),
                "status": "objective_reached" if objective_achieved else "moving_forward" if progress_delta > 0 else "stalled",
            },
            simulator_notes=[
                f"alignment={alignment:.3f}",
                f"keyword_bonus={keyword_bonus:.3f}",
                f"objective_achieved={objective_achieved}",
            ],
            proxy_signals={
                "alignment_proxy": round(alignment, 4),
                "trust_delta": round(trust_delta, 4),
                "frustration_delta": round(frustration_delta, 4),
            },
            objective_achieved=objective_achieved,
        )

    def _build_user_message(
        self,
        sim_input: SimulatorInput,
        alignment: float,
        progress_delta: float,
        objective_achieved: bool,
    ) -> str:
        persona = sim_input.persona
        action = sim_input.latest_action.normalized()
        if objective_achieved:
            return "Yes, this is close to what I needed. I can use this version."
        if persona.preferred_length == "short" and action.verbosity == "high":
            return "This is useful, but can you make it tighter and more direct?"
        if persona.preferred_length == "long" and action.verbosity == "low":
            return "I need a little more detail before I can actually use it."
        if persona.warmth_preference == "high" and action.warmth == "low":
            return "Can you make it feel a bit more understanding without losing the point?"
        if persona.formality_preference == "formal" and action.formality == "low":
            return "I need the tone to read more professional."
        if persona.directness_preference == "high" and action.directness == "low":
            return "Please be more explicit about what I should say or do next."
        if progress_delta > 0.2:
            return "That is getting closer. Can you sharpen it one more step?"
        if alignment > 0.65:
            return "This direction makes sense. Keep going and tighten the wording."
        return "I am not fully there yet. Please adjust based on what seems to matter most to me."

