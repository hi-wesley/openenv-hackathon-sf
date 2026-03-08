from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dfa_agent_env.baselines import BASELINE_REGISTRY, run_baseline
from dfa_agent_env.config import LOGS_DIR
from dfa_agent_env.models import AssistantAction, DFAObservation, EpisodeTrace
from dfa_agent_env.prompts import ASSISTANT_SYSTEM_PROMPT
from dfa_agent_env.scenario_schema import iter_scenarios, load_demo_story_cards
from dfa_agent_env.server.environment import DFAAgentEnvironment
from dfa_agent_env.server.local_hf import generate_chat_text
from dfa_agent_env.server.trace import export_trace_json
from dfa_agent_env.server.utils import list_trace_files
from dfa_agent_env.serialization import parse_action_response


def _render_transcript(observation: DFAObservation) -> str:
    bubbles = []
    for message in observation.conversation:
        role_cls = "assistant" if message.role == "assistant" else "user"
        bubbles.append(
            f"<div class='dfa-bubble {role_cls}'><div class='dfa-role'>{message.role.title()}</div><div>{message.content}</div></div>"
        )
    return "<div class='dfa-transcript'>" + "".join(bubbles) + "</div>"


def _render_emotion_table(trace: dict[str, Any]) -> str:
    logs = trace.get("turn_logs", [])
    if not logs:
        return "<p>No customer replies yet.</p>"
    rows = [
        "<table class='dfa-table'><tr><th>Turn</th><th>Satisfaction</th><th>Happiness</th><th>Anger</th><th>Annoyance</th><th>Gratitude</th></tr>"
    ]
    for log in logs:
        emotions = log["customer_emotion_scores"]
        rows.append(
            "<tr>"
            f"<td>{log['turn_index']}</td>"
            f"<td>{log['customer_satisfaction_score']:.3f}</td>"
            f"<td>{emotions['happiness']:.3f}</td>"
            f"<td>{emotions['anger']:.3f}</td>"
            f"<td>{emotions['annoyance']:.3f}</td>"
            f"<td>{emotions['gratitude']:.3f}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _render_reward_table(trace: dict[str, Any]) -> str:
    logs = trace.get("turn_logs", [])
    if not logs:
        return "<p>No reward breakdown yet.</p>"
    rows = [
        "<table class='dfa-table'><tr><th>Turn</th><th>Total</th><th>Score Delta</th><th>Turn Satisfaction</th><th>Validity</th></tr>"
    ]
    for log in logs:
        reward = log["reward_components"]
        rows.append(
            "<tr>"
            f"<td>{log['turn_index']}</td>"
            f"<td>{reward['combined_reward']:.3f}</td>"
            f"<td>{reward['score_delta_reward']:.3f}</td>"
            f"<td>{reward['turn_satisfaction_reward']:.3f}</td>"
            f"<td>{reward['format_validity_reward']:.3f}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _render_story_cards() -> list[list[str]]:
    cards = []
    for row in load_demo_story_cards():
        cards.append([row["title"], row["scenario_id"], row["family"], row["hook"]])
    return cards


def _live_trace(env: DFAAgentEnvironment) -> dict[str, Any]:
    return {
        "episode_id": env.state.episode_id,
        "scenario": env.state.scenario.model_dump() if env.state.scenario else {},
        "conversation": [message.model_dump() for message in env.state.conversation],
        "turn_logs": [log.model_dump() for log in env.state.per_turn_logs],
        "reward_components": [reward.model_dump() for reward in env.state.reward_components],
        "scorer_result": env.state.satisfaction_score.model_dump() if env.state.satisfaction_score else {},
        "final_summary": env.state.final_summary,
    }


def _done_text(env: DFAAgentEnvironment, observation: DFAObservation) -> str:
    backend_error = env.state.final_summary.get("backend_error")
    if observation.done_reason and backend_error:
        return f"{observation.done_reason}: {backend_error}"
    return observation.done_reason or backend_error or ""


def _diagnostics_payload(env: DFAAgentEnvironment, observation: DFAObservation) -> dict[str, Any]:
    return {
        "error_source": env.state.final_summary.get("last_error_source"),
        "assistant_parse_error": env.state.final_summary.get("last_parse_error"),
        "simulator_error": env.state.final_summary.get("backend_error"),
        "raw_assistant_output": env.state.final_summary.get("last_model_output"),
        "raw_simulator_output": env.state.final_summary.get("last_simulator_output"),
        "observation_parse_error": observation.parse_error,
    }


def build_custom_tab(web_manager, metadata=None, **kwargs):  # pragma: no cover - exercised in runtime
    import gradio as gr

    scenario_rows = list(iter_scenarios())
    families = sorted({row.family for row in scenario_rows})
    initial_family = families[0]
    initial_scenarios = [row.scenario_id for row in scenario_rows if row.family == initial_family]

    css = """
    .dfa-shell {background: linear-gradient(135deg, #f7f3ec 0%, #eef6ff 100%); padding: 12px; border-radius: 18px;}
    .dfa-bubble {padding: 10px 12px; border-radius: 14px; margin: 10px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05);}
    .dfa-bubble.user {background: #fff8ee;}
    .dfa-bubble.assistant {background: #eef5ff;}
    .dfa-role {font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #5e6a7d; margin-bottom: 6px;}
    .dfa-table {width: 100%; border-collapse: collapse; font-size: 12px;}
    .dfa-table th, .dfa-table td {border: 1px solid #d8e0ee; padding: 6px; text-align: center;}
    """

    with gr.Blocks(css=css) as demo:
        env_state = gr.State(None)
        trace_state = gr.State({})
        gr.Markdown("## DFA Agent Dashboard")
        with gr.Row(elem_classes=["dfa-shell"]):
            with gr.Column(scale=3):
                transcript_html = gr.HTML(label="Transcript")
                emotions_html = gr.HTML(label="Customer Emotion Scores")
                reward_html = gr.HTML(label="Reward Breakdown")
            with gr.Column(scale=2):
                family_dropdown = gr.Dropdown(choices=families, value=initial_family, label="Scenario Family")
                scenario_dropdown = gr.Dropdown(choices=initial_scenarios, value=initial_scenarios[0], label="Scenario")
                simulator_dropdown = gr.Dropdown(
                    choices=["local_hf", "mock", "openai_compatible"],
                    value="local_hf",
                    label="Simulator Backend",
                )
                max_turns_slider = gr.Slider(minimum=3, maximum=6, step=1, value=4, label="Max Turns")
                seed_input = gr.Number(value=7, label="Seed", precision=0)
                assistant_backend_dropdown = gr.Dropdown(
                    choices=["local_hf", "baseline"],
                    value="local_hf",
                    label="Assistant Backend",
                )
                baseline_dropdown = gr.Dropdown(
                    choices=list(BASELINE_REGISTRY.keys()),
                    value="default_policy",
                    label="Baseline Policy",
                )
                reset_button = gr.Button("Reset Episode", variant="primary")
                step_button = gr.Button("Step Episode")
                run_full_button = gr.Button("Run Full Episode")
                compare_button = gr.Button("Compare Baselines")
                done_reason = gr.Textbox(label="Done Reason", interactive=False)
                scorer_box = gr.JSON(label="Scorer Result")
            with gr.Column(scale=2):
                metrics_json = gr.JSON(label="Episode Metrics")
                progress_json = gr.JSON(label="Visible Progress")
                diagnostics_json = gr.JSON(label="Diagnostics")
                trace_json = gr.JSON(label="Trace JSON")
                download_file = gr.File(label="Download Trace")

        gr.Markdown("### Baseline Comparison")
        compare_html = gr.HTML()

        gr.Markdown("### Sample Story Cards")
        story_buttons = []
        with gr.Row():
            for card in load_demo_story_cards():
                story_buttons.append(gr.Button(card["title"]))
        story_cards = gr.Dataframe(
            headers=["title", "scenario_id", "family", "hook"],
            value=_render_story_cards(),
            interactive=False,
            wrap=True,
        )

        gr.Markdown("### Trace Comparator")
        with gr.Row():
            trace_file_a = gr.Dropdown(choices=list_trace_files(), label="Trace A")
            trace_file_b = gr.Dropdown(choices=list_trace_files(), label="Trace B")
            load_compare = gr.Button("Load Comparison")
        trace_compare_html = gr.HTML()

        def update_scenario_choices(family: str):
            choices = [row.scenario_id for row in scenario_rows if row.family == family]
            value = choices[0] if choices else None
            return gr.update(choices=choices, value=value)

        def _button_updates(observation: DFAObservation):
            if observation.done:
                return (
                    gr.update(value="Episode Complete", interactive=False),
                    gr.update(value="Episode Complete", interactive=False),
                )
            return (
                gr.update(value="Step Episode", interactive=True),
                gr.update(value="Run Full Episode", interactive=True),
            )

        def _render_episode_outputs(env: DFAAgentEnvironment, observation: DFAObservation, trace: dict[str, Any] | None = None):
            live_trace = trace or env.state.final_summary.get("trace", _live_trace(env))
            download = _maybe_write_trace(env) if observation.done else None
            step_update, run_full_update = _button_updates(observation)
            return (
                env,
                live_trace,
                _render_transcript(observation),
                _render_emotion_table(live_trace),
                _render_reward_table(live_trace),
                observation.episode_metrics_visible,
                observation.task_progress_visible,
                _done_text(env, observation),
                env.state.satisfaction_score.model_dump() if env.state.satisfaction_score else {},
                _diagnostics_payload(env, observation),
                live_trace,
                download,
                step_update,
                run_full_update,
            )

        def _next_action(env: DFAAgentEnvironment, observation: DFAObservation, assistant_backend: str, baseline: str):
            parse_error_override = None
            raw_model_output = None
            if assistant_backend == "local_hf":
                action, parse_error_override, raw_model_output = _local_assistant_action(env, observation)
            else:
                action = run_baseline(baseline, observation)
            return action, parse_error_override, raw_model_output

        def reset_episode(family: str, scenario_id: str, simulator_backend: str, max_turns: int, seed: int):
            env = DFAAgentEnvironment()
            observation = env.reset(
                split="train",
                family=family,
                scenario_id=scenario_id,
                simulator_backend=simulator_backend,
                max_turns=int(max_turns),
                seed=int(seed),
                mode="demo",
            )
            return _render_episode_outputs(env, observation, _live_trace(env))

        def _local_assistant_action(env: DFAAgentEnvironment, observation: DFAObservation):
            try:
                text = generate_chat_text(
                    system_prompt=ASSISTANT_SYSTEM_PROMPT,
                    user_prompt=observation.prompt_text,
                    temperature=env.config.local_assistant_temperature,
                    max_new_tokens=env.config.local_model_max_new_tokens,
                    config=env.config,
                )
                parsed = parse_action_response(text)
                return parsed.action or AssistantAction.default(), parsed.parse_error, parsed.raw_text
            except Exception as exc:
                return AssistantAction.default(), f"Local assistant backend error: {exc}", None

        def step_episode(env: DFAAgentEnvironment, trace: dict, assistant_backend: str, baseline: str):
            if env is None:
                env = DFAAgentEnvironment()
                observation = env.reset(mode="demo")
            else:
                observation = env.current_observation()
            if observation.done:
                return _render_episode_outputs(env, observation, trace or _live_trace(env))
            action, parse_error_override, raw_model_output = _next_action(env, observation, assistant_backend, baseline)
            next_observation = env.step(
                action,
                parse_error_override=parse_error_override,
                raw_model_output=raw_model_output,
            )
            return _render_episode_outputs(env, next_observation)

        def run_full_episode(env: DFAAgentEnvironment | None, trace: dict, assistant_backend: str, baseline: str):
            if env is None:
                env = DFAAgentEnvironment()
                observation = env.reset(mode="demo")
            else:
                observation = env.current_observation()
            if observation.done:
                return _render_episode_outputs(env, observation, trace or _live_trace(env))
            max_iterations = max(1, env.state.max_turns - env.state.turn_index + 2)
            for _ in range(max_iterations):
                if observation.done:
                    break
                action, parse_error_override, raw_model_output = _next_action(env, observation, assistant_backend, baseline)
                observation = env.step(
                    action,
                    parse_error_override=parse_error_override,
                    raw_model_output=raw_model_output,
                )
                if observation.done:
                    break
            return _render_episode_outputs(env, observation)

        def compare_baselines(family: str, scenario_id: str, simulator_backend: str, max_turns: int, seed: int):
            rows = [
                "<table class='dfa-table'><tr><th>Policy</th><th>Total Reward</th><th>Final Satisfaction</th><th>Turns</th><th>Done Reason</th></tr>"
            ]
            for policy_name in BASELINE_REGISTRY:
                env = DFAAgentEnvironment()
                obs = env.reset(
                    split="train",
                    family=family,
                    scenario_id=scenario_id,
                    simulator_backend=simulator_backend,
                    max_turns=int(max_turns),
                    seed=int(seed),
                    mode="demo",
                )
                while not obs.done:
                    obs = env.step(run_baseline(policy_name, obs))
                metrics = env.state.final_summary
                rows.append(
                    "<tr>"
                    f"<td>{policy_name}</td>"
                    f"<td>{metrics.get('total_reward', 0.0):.3f}</td>"
                    f"<td>{metrics.get('customer_satisfaction_score', 0.0):.3f}</td>"
                    f"<td>{metrics.get('turns_used', 0)}</td>"
                    f"<td>{metrics.get('done_reason', '')}</td>"
                    "</tr>"
                )
            rows.append("</table>")
            return "".join(rows)

        def load_trace_pair(path_a: str, path_b: str):
            if not path_a or not path_b:
                return "<p>Select two trace files.</p>"
            trace_a = json.loads(Path(path_a).read_text(encoding="utf-8"))
            trace_b = json.loads(Path(path_b).read_text(encoding="utf-8"))
            score_a = trace_a.get("scorer_result", {}).get("score")
            score_b = trace_b.get("scorer_result", {}).get("score")
            reward_a = sum(item["combined_reward"] for item in trace_a.get("reward_components", []))
            reward_b = sum(item["combined_reward"] for item in trace_b.get("reward_components", []))
            return (
                "<div class='dfa-shell'><div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;'>"
                f"<div><h4>Trace A</h4><p>Scenario: {trace_a.get('scenario', {}).get('scenario_id')}</p><p>Total Reward: {reward_a:.3f}</p><p>Scorer: {score_a}</p></div>"
                f"<div><h4>Trace B</h4><p>Scenario: {trace_b.get('scenario', {}).get('scenario_id')}</p><p>Total Reward: {reward_b:.3f}</p><p>Scorer: {score_b}</p></div>"
                "</div></div>"
            )

        family_dropdown.change(update_scenario_choices, inputs=family_dropdown, outputs=scenario_dropdown)
        reset_button.click(
            reset_episode,
            inputs=[family_dropdown, scenario_dropdown, simulator_dropdown, max_turns_slider, seed_input],
            outputs=[env_state, trace_state, transcript_html, emotions_html, reward_html, metrics_json, progress_json, done_reason, scorer_box, diagnostics_json, trace_json, download_file, step_button, run_full_button],
        )
        for button, card in zip(story_buttons, load_demo_story_cards()):
            button.click(
                lambda simulator_backend, max_turns, seed, family=card["family"], scenario_id=card["scenario_id"]: reset_episode(
                    family,
                    scenario_id,
                    simulator_backend,
                    max_turns,
                    seed,
                ),
                inputs=[simulator_dropdown, max_turns_slider, seed_input],
                outputs=[env_state, trace_state, transcript_html, emotions_html, reward_html, metrics_json, progress_json, done_reason, scorer_box, diagnostics_json, trace_json, download_file, step_button, run_full_button],
            )
        step_button.click(
            step_episode,
            inputs=[env_state, trace_state, assistant_backend_dropdown, baseline_dropdown],
            outputs=[env_state, trace_state, transcript_html, emotions_html, reward_html, metrics_json, progress_json, done_reason, scorer_box, diagnostics_json, trace_json, download_file, step_button, run_full_button],
        )
        run_full_button.click(
            run_full_episode,
            inputs=[env_state, trace_state, assistant_backend_dropdown, baseline_dropdown],
            outputs=[env_state, trace_state, transcript_html, emotions_html, reward_html, metrics_json, progress_json, done_reason, scorer_box, diagnostics_json, trace_json, download_file, step_button, run_full_button],
        )
        compare_button.click(
            compare_baselines,
            inputs=[family_dropdown, scenario_dropdown, simulator_dropdown, max_turns_slider, seed_input],
            outputs=compare_html,
        )
        load_compare.click(load_trace_pair, inputs=[trace_file_a, trace_file_b], outputs=trace_compare_html)

    return demo


def _maybe_write_trace(env: DFAAgentEnvironment):
    trace = env.state.final_summary.get("trace")
    if not trace:
        return None
    path = LOGS_DIR / f"{env.state.episode_id}.json"
    export_trace_json(path, EpisodeTrace(**trace))
    return str(path)
