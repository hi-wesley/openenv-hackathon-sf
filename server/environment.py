from __future__ import annotations

import uuid
from typing import Any, Dict

from dfa_agent_env.compat import Environment
from dfa_agent_env.config import EnvConfig, get_config
from dfa_agent_env.models import (
    AssistantAction,
    ChatState,
    ConversationMessage,
    DFAEnvState,
    DFAObservation,
    EmotionScores,
    ScorerInputs,
    SimulatorInput,
)
from dfa_agent_env.scenario_schema import select_scenario
from dfa_agent_env.scoring import BaseSatisfactionScorer, build_scorer
from dfa_agent_env.serialization import build_prompt_text
from dfa_agent_env.server.persona_sampler import initial_chat_state, update_chat_state
from dfa_agent_env.server.reward_pipeline import compute_reward_components
from dfa_agent_env.server.simulator_factory import build_simulator
from dfa_agent_env.server.trace import build_episode_trace
from dfa_agent_env.server.utils import merge_error_messages, score_customer_emotions, validate_assistant_message


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
        chat_state = self._state.chat_state or ChatState()
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
            customer_emotion_scores=chat_state.emotion_scores,
            customer_satisfaction_score=chat_state.satisfaction_score,
            done_reason=self._state.done_reason,
            episode_metrics_visible=self._episode_metrics(),
            parse_error=self._state.final_summary.get("last_parse_error"),
            simulator_backend=self._state.simulator_backend,
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
        self._scorer = build_scorer(self.config.default_scorer, self.config.constant_scorer_value)
        self._state = DFAEnvState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            scenario=scenario,
            chat_state=initial_chat_state(),
            conversation=[],
            turn_index=0,
            max_turns=max_turns,
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
        )
        self._generate_opening_customer_turn()
        return self.current_observation()

    def step(
        self,
        action: AssistantAction,
        timeout_s: float | None = None,
        *,
        parse_error_override: str | None = None,
        raw_model_output: str | None = None,
        **_: Any,
    ) -> DFAObservation:
        if not self._state.scenario or not self._state.chat_state:
            raise RuntimeError("Call reset() before step().")
        if self._state.done_reason:
            return self.current_observation()

        normalized_action = action.normalized()
        parse_error = merge_error_messages(
            parse_error_override,
            normalized_action.validate_action(self.config.message_char_budget),
            validate_assistant_message(normalized_action.message),
        )
        if parse_error:
            self._state.invalid_action_count += 1

        assistant_message = normalized_action.message
        if parse_error and raw_model_output and raw_model_output.strip():
            assistant_message = raw_model_output.strip()
            self._state.final_summary["last_model_output"] = assistant_message
            self._state.final_summary["last_error_source"] = "assistant"
        elif parse_error:
            self._state.final_summary["last_error_source"] = "assistant"
            self._state.final_summary["last_model_output"] = assistant_message
        elif assistant_message:
            self._state.final_summary["last_model_output"] = assistant_message

        self._state.conversation.append(
            ConversationMessage(
                role="assistant",
                content=assistant_message,
                turn_index=self._state.turn_index + 1,
                metadata={
                    **normalized_action.summary(),
                    "parse_error": parse_error,
                    "raw_model_output_present": bool(raw_model_output and raw_model_output.strip()),
                },
            )
        )

        previous_emotions = self._state.chat_state.emotion_scores
        simulator = build_simulator(self._state.simulator_backend, self.config)
        simulator_output = simulator.generate_reply(
            self._build_simulator_input(normalized_action, assistant_message_override=assistant_message)
        )
        if simulator_output.backend_error:
            self._state.final_summary["backend_error"] = simulator_output.backend_error
            self._state.final_summary["last_error_source"] = "simulator"
        if simulator_output.raw_model_output:
            self._state.final_summary["last_simulator_output"] = simulator_output.raw_model_output

        current_emotions = score_customer_emotions(simulator_output.user_message)
        current_score = current_emotions.composite()
        self._state.chat_state = update_chat_state(
            self._state.chat_state,
            customer_emotions=current_emotions,
            objective_achieved=simulator_output.objective_achieved,
            continue_episode=simulator_output.continue_episode,
            extra_signals=simulator_output.proxy_signals,
        )
        if simulator_output.user_message:
            self._state.conversation.append(
                ConversationMessage(
                    role="user",
                    content=simulator_output.user_message,
                    turn_index=self._state.turn_index + 1,
                    metadata={
                        "simulator_backend": self._state.simulator_backend,
                        "emotion_scores": current_emotions.model_dump(),
                        "satisfaction_score": current_score,
                    },
                )
            )

        self._state.turn_index += 1
        self._state.step_count = self._state.turn_index
        visible_progress = simulator_output.visible_progress_update or {}
        done_reason = self._decide_done(simulator_output, parse_error=parse_error)

        reward_components = compute_reward_components(
            previous_emotions=previous_emotions,
            current_emotions=current_emotions,
            assistant_valid=parse_error is None,
            simulator_valid=simulator_output.backend_error is None,
            parse_error=parse_error,
            simulator_error=simulator_output.backend_error,
            scorer_result=None,
            config=self.config,
        )
        self._state.reward_components.append(reward_components)
        self._state.per_turn_logs.append(
            self._build_turn_log(
                assistant_message,
                simulator_output.user_message,
                current_emotions,
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

        self._state.final_summary["last_parse_error"] = parse_error
        self._state.final_summary["last_reward"] = reward_components.combined_reward
        observation = self.current_observation()
        observation.reward = reward_components.combined_reward
        observation.done = bool(done_reason)
        observation.done_reason = done_reason
        return observation

    def close(self) -> None:
        return None

    def _generate_opening_customer_turn(self) -> None:
        simulator = build_simulator(self._state.simulator_backend, self.config)
        output = simulator.generate_opening_message(self._build_simulator_input(None, opening=True))
        if output.backend_error:
            self._state.final_summary["backend_error"] = output.backend_error
            self._state.final_summary["last_error_source"] = "simulator"
        if output.raw_model_output:
            self._state.final_summary["last_simulator_output"] = output.raw_model_output
        opening_text = (output.user_message or "").strip()
        if not opening_text:
            self._state.done_reason = "fatal_backend_error" if output.backend_error else "empty_opening_message"
            self._state.final_summary = self._episode_metrics()
            self._state.final_summary["backend_error"] = output.backend_error or "Simulator returned an empty opening message."
            self._state.final_summary["last_error_source"] = "simulator"
            self._state.final_summary["trace"] = build_episode_trace(self._state).model_dump()
            return
        opening_emotions = score_customer_emotions(opening_text)
        opening_score = opening_emotions.composite()
        self._state.chat_state = update_chat_state(
            self._state.chat_state or initial_chat_state(),
            customer_emotions=opening_emotions,
            objective_achieved=output.objective_achieved,
            continue_episode=output.continue_episode,
            extra_signals=output.proxy_signals,
        )
        self._state.conversation.append(
            ConversationMessage(
                role="user",
                content=opening_text,
                turn_index=0,
                metadata={
                    "simulator_backend": self._state.simulator_backend,
                    "opening_turn": True,
                    "emotion_scores": opening_emotions.model_dump(),
                    "satisfaction_score": opening_score,
                },
            )
        )
    def _build_simulator_input(
        self,
        action: AssistantAction | None,
        opening: bool = False,
        assistant_message_override: str | None = None,
    ) -> SimulatorInput:
        return SimulatorInput(
            scenario=self._state.scenario,
            conversation=self._state.conversation,
            latest_assistant_message=assistant_message_override if assistant_message_override is not None else (action.message if action else ""),
            latest_customer_message=self._latest_user_message(),
            latest_customer_emotions=(
                self._state.chat_state.emotion_scores if self._state.chat_state else EmotionScores()
            ),
            turn_index=self._state.turn_index + (0 if opening else 1),
            max_turns=self._state.max_turns,
            mode=self._state.mode,
            opening_turn=opening,
        )

    def _decide_done(self, simulator_output, *, parse_error: str | None) -> str | None:
        if parse_error and self._state.invalid_action_count >= self.config.invalid_action_threshold:
            return "invalid_action_threshold_reached"
        if simulator_output.backend_error and self.config.strict_backend_errors:
            return "fatal_backend_error"
        if simulator_output.objective_achieved or (self._state.chat_state and self._state.chat_state.objective_achieved):
            return "objective_achieved"
        if not simulator_output.continue_episode:
            return "simulator_stopped"
        if self._state.turn_index >= self._state.max_turns:
            return "max_turns_reached"
        return None

    def _finalize_episode(self):
        self._state.scorer_inputs = ScorerInputs(
            scenario=self._state.scenario,
            conversation=self._state.conversation,
            turn_logs=self._state.per_turn_logs,
            reward_components=self._state.reward_components,
            final_chat_state=self._state.chat_state,
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
        reward.final_satisfaction_reward = final_score
        reward.combined_reward = (
            self.config.reward_weight_score_delta * reward.score_delta_reward
            + self.config.reward_weight_turn_satisfaction * reward.turn_satisfaction_reward
            + self.config.reward_weight_format_validity * reward.format_validity_reward
            + self.config.reward_weight_satisfaction_score * reward.final_satisfaction_reward
        )
        if self._state.per_turn_logs:
            self._state.per_turn_logs[-1].reward_components = reward

    def _build_turn_log(
        self,
        assistant_message: str,
        customer_message: str,
        customer_emotions: EmotionScores,
        reward_components,
        parse_error: str | None,
        visible_progress: Dict[str, Any],
        simulator_output,
        done_reason: str | None,
    ):
        from dfa_agent_env.models import TurnLog

        return TurnLog(
            turn_index=self._state.turn_index,
            assistant_message=assistant_message,
            customer_message=customer_message,
            customer_emotion_scores=customer_emotions,
            customer_satisfaction_score=customer_emotions.composite(),
            reward_components=reward_components,
            parse_valid=parse_error is None and simulator_output.backend_error is None,
            parse_error=parse_error,
            visible_progress=visible_progress,
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
            if message.role == "assistant":
                return {
                    "message_length": len(message.content),
                }
        return {}

    def _visible_progress(self) -> Dict[str, Any]:
        if self._state.per_turn_logs:
            latest = self._state.per_turn_logs[-1]
            return {
                "status": latest.visible_progress.get("status", "conversation_active"),
                "current_satisfaction_score": latest.customer_satisfaction_score,
                "emotion_scores": latest.customer_emotion_scores.model_dump(),
            }
        chat_state = self._state.chat_state or ChatState()
        return {
            "status": "customer_opened_case",
            "current_satisfaction_score": chat_state.satisfaction_score,
            "emotion_scores": chat_state.emotion_scores.model_dump(),
        }

    def _episode_metrics(self) -> Dict[str, Any]:
        total_reward = sum(item.combined_reward for item in self._state.reward_components)
        shaped_reward = sum(
            item.score_delta_reward + item.turn_satisfaction_reward + item.format_validity_reward
            for item in self._state.reward_components
        )
        final_reward = (
            self._state.satisfaction_score.score
            if self._state.satisfaction_score and self._state.satisfaction_score.score is not None
            else 0.0
        )
        chat_state = self._state.chat_state or ChatState()
        return {
            "total_reward": round(total_reward, 4),
            "shaped_reward_only": round(shaped_reward, 4),
            "final_satisfaction_reward": round(float(final_reward), 4),
            "task_completion_flag": bool(chat_state.objective_achieved),
            "invalid_action_count": self._state.invalid_action_count,
            "turns_used": self._state.turn_index,
            "early_termination": self._state.turn_index < self._state.max_turns and bool(self._state.done_reason),
            "scenario_family": self._state.scenario.family if self._state.scenario else None,
            "simulator_backend": self._state.simulator_backend,
            "done_reason": self._state.done_reason,
            "backend_error": self._state.final_summary.get("backend_error"),
            "last_model_output": self._state.final_summary.get("last_model_output"),
            "last_simulator_output": self._state.final_summary.get("last_simulator_output"),
            "last_parse_error": self._state.final_summary.get("last_parse_error"),
            "last_error_source": self._state.final_summary.get("last_error_source"),
            "customer_emotion_scores": chat_state.emotion_scores.model_dump(),
            "customer_satisfaction_score": round(float(chat_state.satisfaction_score), 4),
            "customer_summary": {
                "emotion_scores": chat_state.emotion_scores.model_dump(),
                "satisfaction_score": round(float(chat_state.satisfaction_score), 4),
                "objective_achieved": bool(chat_state.objective_achieved),
            },
        }
