from __future__ import annotations

import uuid
from typing import Any, Dict

from dfa_agent_env.compat import Environment
from dfa_agent_env.config import EnvConfig, get_config
from dfa_agent_env.models import (
    AssistantAction,
    ConversationMessage,
    DFAEnvState,
    DFAObservation,
    ParseOutcome,
    ScorerInputs,
)
from dfa_agent_env.scenario_schema import select_scenario
from dfa_agent_env.scoring import BaseSatisfactionScorer, build_scorer
from dfa_agent_env.serialization import build_prompt_text
from dfa_agent_env.server.persona_sampler import (
    apply_assistant_action,
    apply_simulator_delta,
    initial_hidden_state,
    sample_persona,
)
from dfa_agent_env.server.reward_pipeline import compute_reward_components
from dfa_agent_env.server.simulator_factory import build_simulator
from dfa_agent_env.server.trace import build_episode_trace


class DFAAgentEnvironment(Environment[AssistantAction, DFAObservation, DFAEnvState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        *,
        config: EnvConfig | None = None,
        scorer: BaseSatisfactionScorer | None = None,
    ) -> None:
        self.config = config or get_config()
        self._scorer = scorer or build_scorer(self.config.default_scorer, self.config.constant_scorer_value)
        self._state = DFAEnvState()

    @property
    def state(self) -> DFAEnvState:
        return self._state

    def current_observation(self) -> DFAObservation:
        if not self._state.scenario:
            raise RuntimeError("Environment has not been reset.")
        observation = DFAObservation(
            scenario_id=self._state.scenario.scenario_id,
            family=self._state.scenario.family,
            turn_index=self._state.turn_index,
            max_turns=self._state.max_turns,
            conversation=self._state.conversation,
            latest_user_message=self._latest_user_message(),
            visible_context=self._state.scenario.visible_context,
            assistant_last_action_summary=self._last_action_summary(),
            task_progress_visible=self._visible_progress(),
            done_reason=self._state.done_reason,
            available_style_axes=[
                "verbosity",
                "warmth",
                "humor",
                "formality",
                "directness",
                "initiative",
                "explanation_depth",
                "acknowledgement_style",
            ],
            episode_metrics_visible=self._episode_metrics(),
            parse_error=self._state.final_summary.get("last_parse_error"),
            simulator_backend=self._state.simulator_backend,
            revealable_persona=self._reveal_persona_if_allowed(),
            done=bool(self._state.done_reason),
            reward=self._state.reward_components[-1].combined_reward if self._state.reward_components else 0.0,
        )
        observation.prompt_text = build_prompt_text(
            observation,
            template_name="rich_demo" if self._state.mode == "demo" else "compact_train",
        )
        return observation

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        *,
        split: str = "train",
        scenario_id: str | None = None,
        max_turns: int | None = None,
        mode: str = "demo",
        simulator_backend: str | None = None,
        difficulty: str | None = None,
        reveal_persona_after_done: bool | None = None,
        use_debug_heuristic_scorer: bool = False,
        family: str | None = None,
        **_: Any,
    ) -> DFAObservation:
        scenario = select_scenario(
            split=split,
            family=family,
            difficulty=difficulty,
            scenario_id=scenario_id,
            seed=seed,
        )
        max_turns = max_turns or (
            self.config.default_max_turns_demo if mode == "demo" else self.config.default_max_turns_train
        )
        max_turns = min(max_turns, self.config.max_turns_limit)
        rng_seed = seed if seed is not None else 0
        rng = __import__("random").Random(rng_seed)
        persona = sample_persona(rng, scenario, difficulty=difficulty)
        hidden_state = initial_hidden_state(persona, scenario)
        if use_debug_heuristic_scorer:
            self._scorer = build_scorer("debug_proxy", self.config.constant_scorer_value)
        else:
            self._scorer = build_scorer(self.config.default_scorer, self.config.constant_scorer_value)
        self._state = DFAEnvState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            scenario=scenario,
            persona=persona,
            hidden_state=hidden_state,
            conversation=[
                ConversationMessage(
                    role="user",
                    content=scenario.initial_user_message,
                    turn_index=0,
                    metadata={"source": "scenario"},
                )
            ],
            turn_index=0,
            max_turns=max_turns,
            task_progress_hidden={"goal_progress": 0.0, "task_completed": False},
            satisfaction_score=None,
            per_turn_logs=[],
            final_summary={},
            reward_components=[],
            simulator_trace=[],
            scorer_inputs=None,
            seed=seed,
            mode=mode,
            invalid_action_count=0,
            done_reason=None,
            simulator_backend=(simulator_backend or self.config.default_simulator_backend),
            reveal_persona_after_done=(
                self.config.reveal_persona_after_done_default
                if reveal_persona_after_done is None
                else reveal_persona_after_done
            ),
        )
        return self.current_observation()

    def step(self, action: AssistantAction, timeout_s: float | None = None, **_: Any) -> DFAObservation:
        if not self._state.scenario or not self._state.persona or not self._state.hidden_state:
            raise RuntimeError("Call reset() before step().")
        if self._state.done_reason:
            return self.current_observation()
        normalized_action = action.normalized()
        parse_error = None
        validation_errors = normalized_action.validate_action(self.config.message_char_budget)
        if validation_errors:
            self._state.invalid_action_count += 1
            parse_error = "; ".join(validation_errors)
            normalized_action = AssistantAction.default("I want to help with this. Let me adjust.")
        self._state.conversation.append(
            ConversationMessage(
                role="assistant",
                content=normalized_action.message,
                turn_index=self._state.turn_index + 1,
                strategy=normalized_action.strategy_summary(),
            )
        )
        self._state.hidden_state = apply_assistant_action(self._state.hidden_state, self._state.persona, normalized_action)
        simulator = build_simulator(self._state.simulator_backend, self.config)
        sim_input = self._build_simulator_input(normalized_action)
        simulator_output = simulator.generate_reply(sim_input)
        if simulator_output.backend_error:
            self._state.final_summary["backend_error"] = simulator_output.backend_error
        self._state.hidden_state = apply_simulator_delta(self._state.hidden_state, simulator_output.latent_state_delta)
        if simulator_output.user_message:
            self._state.conversation.append(
                ConversationMessage(
                    role="user",
                    content=simulator_output.user_message,
                    turn_index=self._state.turn_index + 1,
                    metadata={"simulator_backend": self._state.simulator_backend},
                )
            )
        self._state.turn_index += 1
        self._state.step_count = self._state.turn_index
        visible_progress = simulator_output.visible_progress_update or {}
        self._state.task_progress_hidden["goal_progress"] = self._state.hidden_state.goal_progress
        self._state.task_progress_hidden["task_completed"] = bool(simulator_output.objective_achieved)
        done_reason = self._decide_done(simulator_output, parse_error=parse_error)
        reward_components = compute_reward_components(
            action=normalized_action,
            persona=self._state.persona,
            simulator_output=simulator_output,
            parse_valid=parse_error is None,
            parse_error=parse_error,
            scorer_result=None,
            config=self.config,
        )
        self._state.reward_components.append(reward_components)
        self._state.per_turn_logs.append(
            self._build_turn_log(
                normalized_action,
                simulator_output.user_message,
                reward_components,
                parse_error,
                visible_progress,
                simulator_output,
                done_reason,
            )
        )
        if done_reason:
            self._state.done_reason = done_reason
            scorer_result = self._finalize_episode()
            self._apply_final_scorer_reward(scorer_result)
            trace = build_episode_trace(self._state)
            self._state.final_summary = self._episode_metrics()
            self._state.final_summary["trace"] = trace.model_dump()
            self._state.final_summary["scorer"] = scorer_result.model_dump()
        observation = self.current_observation()
        observation.reward = reward_components.combined_reward
        observation.done = bool(done_reason)
        observation.done_reason = done_reason
        self._state.final_summary["last_parse_error"] = parse_error
        self._state.final_summary["last_reward"] = reward_components.combined_reward
        return observation

    def close(self) -> None:
        return None

    def _build_simulator_input(self, action: AssistantAction):
        from dfa_agent_env.models import SimulatorInput

        return SimulatorInput(
            scenario=self._state.scenario,
            persona=self._state.persona,
            hidden_state=self._state.hidden_state,
            conversation=self._state.conversation,
            latest_action=action,
            latest_assistant_message=action.message,
            turn_index=self._state.turn_index + 1,
            max_turns=self._state.max_turns,
            mode=self._state.mode,
        )

    def _decide_done(self, simulator_output, *, parse_error: str | None) -> str | None:
        if parse_error and self._state.invalid_action_count >= self.config.invalid_action_threshold:
            return "invalid_action_threshold_reached"
        if simulator_output.backend_error and self.config.strict_backend_errors:
            return "fatal_backend_error"
        if simulator_output.objective_achieved:
            return "objective_achieved"
        if not simulator_output.continue_episode:
            return "simulator_stopped"
        if self._state.turn_index >= self._state.max_turns:
            return "max_turns_reached"
        return None

    def _finalize_episode(self):
        self._state.scorer_inputs = ScorerInputs(
            scenario=self._state.scenario,
            persona=self._state.persona,
            conversation=self._state.conversation,
            turn_logs=self._state.per_turn_logs,
            reward_components=self._state.reward_components,
            final_hidden_state=self._state.hidden_state,
            final_summary=self._episode_metrics(),
            mode=self._state.mode,
            simulator_backend=self._state.simulator_backend,
        )
        trace = build_episode_trace(self._state)
        scorer_result = self._scorer.score(trace)
        self._state.satisfaction_score = scorer_result
        return scorer_result

    def _apply_final_scorer_reward(self, scorer_result) -> None:
        if not self._state.reward_components:
            return
        final_score = 0.0
        if scorer_result.available and scorer_result.score is not None:
            final_score = float(scorer_result.score)
        reward = self._state.reward_components[-1]
        reward.satisfaction_score_reward = final_score
        reward.combined_reward = (
            self.config.reward_weight_task_progress * reward.task_progress_reward
            + self.config.reward_weight_format_validity * reward.format_validity_reward
            + self.config.reward_weight_instruction_following * reward.instruction_following_reward
            + self.config.reward_weight_satisfaction_score * reward.satisfaction_score_reward
        )
        if self._state.per_turn_logs:
            self._state.per_turn_logs[-1].reward_components = reward

    def _build_turn_log(
        self,
        action: AssistantAction,
        user_message: str,
        reward_components,
        parse_error: str | None,
        visible_progress: Dict[str, Any],
        simulator_output,
        done_reason: str | None,
    ):
        from dfa_agent_env.models import TurnLog

        return TurnLog(
            turn_index=self._state.turn_index,
            assistant_action=action.model_dump(),
            user_message=user_message,
            reward_components=reward_components,
            parse_valid=parse_error is None,
            parse_error=parse_error,
            visible_progress=visible_progress,
            hidden_state_snapshot=self._state.hidden_state.model_dump(),
            simulator_notes=simulator_output.simulator_notes,
            proxy_signals=simulator_output.proxy_signals,
            done=bool(done_reason),
            done_reason=done_reason,
        )

    def _latest_user_message(self) -> str:
        for message in reversed(self._state.conversation):
            if message.role == "user":
                return message.content
        return ""

    def _last_action_summary(self) -> Dict[str, Any]:
        for message in reversed(self._state.conversation):
            if message.role == "assistant" and message.strategy:
                return message.strategy
        return {}

    def _visible_progress(self) -> Dict[str, Any]:
        if self._state.per_turn_logs:
            return self._state.per_turn_logs[-1].visible_progress
        return {"status": "episode_started", "goal_progress_hint": 0.0}

    def _episode_metrics(self) -> Dict[str, Any]:
        total_reward = sum(item.combined_reward for item in self._state.reward_components)
        shaped_reward = sum(
            item.task_progress_reward + item.format_validity_reward + item.instruction_following_reward
            for item in self._state.reward_components
        )
        final_reward = self._state.satisfaction_score.score if self._state.satisfaction_score and self._state.satisfaction_score.score is not None else 0.0
        return {
            "total_reward": round(total_reward, 4),
            "shaped_reward_only": round(shaped_reward, 4),
            "final_satisfaction_reward": round(float(final_reward), 4),
            "task_completion_flag": bool(self._state.task_progress_hidden.get("task_completed", False)),
            "invalid_action_count": self._state.invalid_action_count,
            "turns_used": self._state.turn_index,
            "early_termination": self._state.turn_index < self._state.max_turns and bool(self._state.done_reason),
            "scenario_family": self._state.scenario.family if self._state.scenario else None,
            "persona_summary": self._state.persona.reveal_dict() if self._state.persona else {},
            "simulator_backend": self._state.simulator_backend,
            "done_reason": self._state.done_reason,
        }

    def _reveal_persona_if_allowed(self):
        if self._state.done_reason and self._state.reveal_persona_after_done and self._state.persona:
            return self._state.persona.reveal_dict()
        return None
