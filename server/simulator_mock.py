from __future__ import annotations

from dfa_agent_env.models import SimulatorInput, SimulatorOutput
from dfa_agent_env.server.simulator_base import BaseUserSimulator
from dfa_agent_env.server.utils import message_quality_score


class MockUserSimulator(BaseUserSimulator):
    name = "mock"

    def generate_opening_message(self, sim_input: SimulatorInput) -> SimulatorOutput:
        scenario = sim_input.scenario
        opener = scenario.initial_user_message.strip()
        if scenario.family == "late_delivery_refund":
            message = f"My order still has not arrived. {opener} I need a real solution, not another vague update."
        elif scenario.family == "damaged_item_replacement":
            message = f"The item arrived damaged. {opener} I need to know how you are going to fix this."
        else:
            message = f"I was charged when I thought this was canceled. {opener} This is extremely frustrating."
        return SimulatorOutput(
            user_message=message,
            continue_episode=True,
            visible_progress_update={"status": "customer_opened_case"},
            simulator_notes=["mock opening message"],
            proxy_signals={"opening_turn": 1.0},
            objective_achieved=False,
        )

    def generate_reply(self, sim_input: SimulatorInput) -> SimulatorOutput:
        score = message_quality_score(sim_input.latest_assistant_message, sim_input.scenario)
        objective_achieved = score >= 0.78
        continue_episode = (sim_input.turn_index < sim_input.max_turns) and not objective_achieved
        user_message = self._build_user_message(sim_input, score, objective_achieved)
        status = "resolved" if objective_achieved else "improving" if score >= 0.55 else "still_upset"
        return SimulatorOutput(
            user_message=user_message,
            continue_episode=continue_episode,
            visible_progress_update={
                "status": status,
                "message_quality_score": round(score, 3),
            },
            simulator_notes=[f"message_quality_score={score:.3f}"],
            proxy_signals={"message_quality_score": round(score, 4)},
            objective_achieved=objective_achieved,
        )

    def _build_user_message(self, sim_input: SimulatorInput, score: float, objective_achieved: bool) -> str:
        family = sim_input.scenario.family
        if objective_achieved:
            if family == "late_delivery_refund":
                return "That helps. If you can actually process the refund or fix the shipment, I’m satisfied with that."
            if family == "damaged_item_replacement":
                return "Okay, that replacement plan works for me. Thanks for actually handling it."
            return "That makes sense and it sounds like you can fix the charge. I appreciate that."
        if score >= 0.55:
            if family == "late_delivery_refund":
                return "This is better than the earlier responses. I’m still annoyed, but at least it sounds like you’re trying to fix it."
            if family == "damaged_item_replacement":
                return "That sounds more helpful. I still need a real replacement, but this is moving in the right direction."
            return "I’m still frustrated about the charge, but this is at least clearer than before."
        if family == "late_delivery_refund":
            return "I’m still waiting and this still feels like a runaround. I need a concrete fix."
        if family == "damaged_item_replacement":
            return "This still does not tell me how you are fixing the damaged item. I need an actual solution."
        return "I’m still frustrated and I still need someone to fix the billing problem instead of circling around it."
