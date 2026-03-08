from __future__ import annotations

import argparse
from pathlib import Path

from dfa_agent_env.baselines import BASELINE_REGISTRY, run_baseline
from dfa_agent_env.models import EpisodeTrace
from dfa_agent_env.scenario_schema import load_demo_story_cards
from dfa_agent_env.server.environment import DFAAgentEnvironment
from dfa_agent_env.server.trace import export_trace_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export curated demo traces.")
    parser.add_argument("--output-dir", default="outputs/logs/demo_traces")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for card in load_demo_story_cards():
        for policy_name in ("default_policy", "always_concise_policy", "always_warm_detailed_policy"):
            env = DFAAgentEnvironment()
            obs = env.reset(
                split="train",
                scenario_id=card["scenario_id"],
                simulator_backend="mock",
                seed=args.seed,
                max_turns=4,
                mode="demo",
                reveal_persona_after_done=True,
            )
            while not obs.done:
                obs = env.step(run_baseline(policy_name, obs))
            trace = EpisodeTrace(**env.state.final_summary["trace"])
            export_trace_json(output_dir / f"{card['scenario_id']}__{policy_name}.json", trace)
    print(f"Exported demo traces to {output_dir}")


if __name__ == "__main__":
    main()
