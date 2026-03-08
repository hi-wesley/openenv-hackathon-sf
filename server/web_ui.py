from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dfa_agent_env.baselines import BASELINE_REGISTRY, run_baseline
from dfa_agent_env.config import LOGS_DIR
from dfa_agent_env.models import DFAObservation, EpisodeTrace
from dfa_agent_env.scenario_schema import iter_scenarios, load_demo_story_cards
from dfa_agent_env.server.environment import DFAAgentEnvironment
from dfa_agent_env.server.trace import export_trace_json
from dfa_agent_env.server.utils import list_trace_files


def _render_transcript(observation: DFAObservation) -> str:
    bubbles = []
    for message in observation.conversation:
        role_cls = "assistant" if message.role == "assistant" else "user"
        bubbles.append(
            f"<div class='dfa-bubble {role_cls}'><div class='dfa-role'>{message.role.title()}</div><div>{message.content}</div></div>"
        )
    return "<div class='dfa-transcript'>" + "".join(bubbles) + "</div>"


def _render_strategy_heatmap(trace: dict[str, Any]) -> str:
    logs = trace.get("turn_logs", [])
    if not logs:
        return "<p>No strategy data yet.</p>"
    axes = ["verbosity", "warmth", "humor", "formality", "directness", "initiative", "explanation_depth"]
    colors = {"low": "#d6e9ff", "medium": "#8bb6ff", "high": "#2357a6"}
    rows = ["<table class='dfa-table'><tr><th>Turn</th>" + "".join(f"<th>{axis}</th>" for axis in axes) + "</tr>"]
    for log in logs:
        action = log["assistant_action"]
        cells = [f"<td>{log['turn_index']}</td>"]
        for axis in axes:
            value = action.get(axis, "medium")
            cells.append(f"<td style='background:{colors.get(value, '#eef4ff')}'>{value}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</table>")
    return "".join(rows)


def _render_reward_table(trace: dict[str, Any]) -> str:
    logs = trace.get("turn_logs", [])
    if not logs:
        return "<p>No reward breakdown yet.</p>"
    rows = [
        "<table class='dfa-table'><tr><th>Turn</th><th>Total</th><th>Progress</th><th>Validity</th><th>Instruction</th><th>Align</th></tr>"
    ]
    for log in logs:
        reward = log["reward_components"]
        rows.append(
            "<tr>"
            f"<td>{log['turn_index']}</td>"
            f"<td>{reward['combined_reward']:.3f}</td>"
            f"<td>{reward['task_progress_reward']:.3f}</td>"
            f"<td>{reward['format_validity_reward']:.3f}</td>"
            f"<td>{reward['instruction_following_reward']:.3f}</td>"
            f"<td>{reward['alignment_proxy']:.3f}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _render_persona(persona: dict[str, Any] | None) -> str:
    if not persona:
        return "<p>Persona remains hidden until the episode ends or reveal is enabled.</p>"
    axis_keys = [
        "preferred_length",
        "warmth_preference",
        "humor_tolerance",
        "formality_preference",
        "directness_preference",
        "initiative_preference",
        "explanation_depth_preference",
    ]
    level_map = {"short": 0.2, "medium": 0.5, "long": 0.8, "none": 0.0, "light": 0.45, "casual": 0.2, "neutral": 0.5, "formal": 0.8, "low": 0.2, "high": 0.8}
    bars = []
    for key in axis_keys:
        raw = persona.get(key, "medium")
        value = level_map.get(raw, 0.5)
        bars.append(
            f"<div class='dfa-bar-row'><span>{key}</span><div class='dfa-bar'><div class='dfa-bar-fill' style='width:{int(value*100)}%'></div></div><span>{raw}</span></div>"
        )
    return "<div>" + "".join(bars) + "</div>"


def _render_story_cards() -> list[list[str]]:
    cards = []
    for row in load_demo_story_cards():
        cards.append([row["title"], row["scenario_id"], row["family"], row["hook"]])
    return cards


def build_custom_tab(web_manager, metadata=None, **kwargs):  # pragma: no cover - exercised in deployed runtime
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
    .dfa-bar-row {display: grid; grid-template-columns: 190px 1fr 80px; gap: 10px; align-items: center; margin: 8px 0;}
    .dfa-bar {height: 14px; background: #dde8f6; border-radius: 999px; overflow: hidden;}
    .dfa-bar-fill {height: 100%; background: linear-gradient(90deg, #214f92, #74a1ff);}
    """

    with gr.Blocks(css=css) as demo:
        env_state = gr.State(None)
        trace_state = gr.State({})
        gr.Markdown("## DFA Agent Dashboard")
        with gr.Row(elem_classes=["dfa-shell"]):
            with gr.Column(scale=3):
                transcript_html = gr.HTML(label="Transcript")
                strategy_html = gr.HTML(label="Strategy Heatmap")
                reward_html = gr.HTML(label="Reward Breakdown")
            with gr.Column(scale=2):
                family_dropdown = gr.Dropdown(choices=families, value=initial_family, label="Scenario Family")
                scenario_dropdown = gr.Dropdown(choices=initial_scenarios, value=initial_scenarios[0], label="Scenario")
                simulator_dropdown = gr.Dropdown(
                    choices=["mock", "openai_compatible"],
                    value="mock",
                    label="Simulator Backend",
                )
                max_turns_slider = gr.Slider(minimum=3, maximum=5, step=1, value=4, label="Max Turns")
                seed_input = gr.Number(value=7, label="Seed", precision=0)
                baseline_dropdown = gr.Dropdown(
                    choices=list(BASELINE_REGISTRY.keys()),
                    value="default_policy",
                    label="Baseline Policy",
                )
                reveal_persona = gr.Checkbox(value=True, label="Reveal Persona At End")
                reset_button = gr.Button("Reset Episode", variant="primary")
                step_button = gr.Button("Step Baseline")
                compare_button = gr.Button("Compare Baselines")
                done_reason = gr.Textbox(label="Done Reason", interactive=False)
                scorer_box = gr.JSON(label="Scorer Result")
            with gr.Column(scale=2):
                persona_html = gr.HTML(label="Persona Radar / Bars")
                metrics_json = gr.JSON(label="Episode Metrics")
                progress_json = gr.JSON(label="Task Progress")
                trace_json = gr.JSON(label="Trace JSON")
                download_file = gr.File(label="Download Trace")

        gr.Markdown("### Before / After Training Comparator")
        with gr.Row():
            trace_file_a = gr.Dropdown(choices=list_trace_files(), label="Trace A")
            trace_file_b = gr.Dropdown(choices=list_trace_files(), label="Trace B")
            load_compare = gr.Button("Load Comparison")
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

        def update_scenario_choices(family: str):
            choices = [row.scenario_id for row in scenario_rows if row.family == family]
            value = choices[0] if choices else None
            return gr.update(choices=choices, value=value)

        def reset_episode(family: str, scenario_id: str, simulator_backend: str, max_turns: int, seed: int, baseline: str, reveal: bool):
            env = DFAAgentEnvironment()
            observation = env.reset(
                split="train",
                family=family,
                scenario_id=scenario_id,
                simulator_backend=simulator_backend,
                max_turns=int(max_turns),
                seed=int(seed),
                mode="demo",
                reveal_persona_after_done=reveal,
            )
            trace = env.state.final_summary.get("trace", {})
            metrics = env.state.final_summary or observation.episode_metrics_visible
            return (
                env,
                trace,
                _render_transcript(observation),
                "<p>No strategy data yet.</p>",
                "<p>No reward breakdown yet.</p>",
                _render_persona(observation.revealable_persona),
                metrics,
                observation.task_progress_visible,
                observation.done_reason or "",
                env.state.satisfaction_score.model_dump() if env.state.satisfaction_score else {},
                trace,
                None,
            )

        def step_episode(env: DFAAgentEnvironment, trace: dict, baseline: str, reveal: bool):
            if env is None:
                env = DFAAgentEnvironment()
                observation = env.reset(mode="demo", reveal_persona_after_done=reveal)
            else:
                observation = env.current_observation()
            if observation.done:
                return (
                    env,
                    trace,
                    _render_transcript(observation),
                    _render_strategy_heatmap(trace or observation.metadata.get("trace", {})),
                    _render_reward_table(trace or observation.metadata.get("trace", {})),
                    _render_persona(observation.revealable_persona),
                    observation.episode_metrics_visible,
                    observation.task_progress_visible,
                    observation.done_reason or "",
                    env.state.satisfaction_score.model_dump() if env.state.satisfaction_score else {},
                    trace,
                    _maybe_write_trace(env),
                )
            action = run_baseline(baseline, observation)
            next_observation = env.step(action)
            new_trace = env.state.final_summary.get("trace", trace)
            return (
                env,
                new_trace,
                _render_transcript(next_observation),
                _render_strategy_heatmap(new_trace),
                _render_reward_table(new_trace),
                _render_persona(next_observation.revealable_persona),
                next_observation.episode_metrics_visible,
                next_observation.task_progress_visible,
                next_observation.done_reason or "",
                env.state.satisfaction_score.model_dump() if env.state.satisfaction_score else {},
                new_trace,
                _maybe_write_trace(env) if next_observation.done else None,
            )

        def compare_baselines(family: str, scenario_id: str, simulator_backend: str, max_turns: int, seed: int):
            rows = [
                "<table class='dfa-table'><tr><th>Policy</th><th>Total Reward</th><th>Turns</th><th>Done Reason</th></tr>"
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
            inputs=[family_dropdown, scenario_dropdown, simulator_dropdown, max_turns_slider, seed_input, baseline_dropdown, reveal_persona],
            outputs=[env_state, trace_state, transcript_html, strategy_html, reward_html, persona_html, metrics_json, progress_json, done_reason, scorer_box, trace_json, download_file],
        )
        for button, card in zip(story_buttons, load_demo_story_cards()):
            button.click(
                lambda simulator_backend, max_turns, seed, baseline, reveal, family=card["family"], scenario_id=card["scenario_id"]: reset_episode(
                    family,
                    scenario_id,
                    simulator_backend,
                    max_turns,
                    seed,
                    baseline,
                    reveal,
                ),
                inputs=[simulator_dropdown, max_turns_slider, seed_input, baseline_dropdown, reveal_persona],
                outputs=[env_state, trace_state, transcript_html, strategy_html, reward_html, persona_html, metrics_json, progress_json, done_reason, scorer_box, trace_json, download_file],
            )
        step_button.click(
            step_episode,
            inputs=[env_state, trace_state, baseline_dropdown, reveal_persona],
            outputs=[env_state, trace_state, transcript_html, strategy_html, reward_html, persona_html, metrics_json, progress_json, done_reason, scorer_box, trace_json, download_file],
        )
        compare_button.click(
            compare_baselines,
            inputs=[family_dropdown, scenario_dropdown, simulator_dropdown, max_turns_slider, seed_input],
            outputs=compare_html,
        )
        load_compare.click(load_trace_pair, inputs=[trace_file_a, trace_file_b], outputs=compare_html)

    return demo


def _maybe_write_trace(env: DFAAgentEnvironment):
    trace = env.state.final_summary.get("trace")
    if not trace:
        return None
    path = LOGS_DIR / f"{env.state.episode_id}.json"
    export_trace_json(path, EpisodeTrace(**trace))
    return str(path)
