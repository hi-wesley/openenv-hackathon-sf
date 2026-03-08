from __future__ import annotations

import argparse

from dfa_agent_env.baselines import run_baseline
from dfa_agent_env.models import EpisodeTrace
from dfa_agent_env.scenario_schema import load_demo_story_cards
from dfa_agent_env.server.environment import DFAAgentEnvironment
from dfa_agent_env.server.trace import export_trace_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a few canned rollouts for demo mode.")
    parser.add_argument("--output-dir", default="outputs/logs/sample_rollouts")
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    for card in load_demo_story_cards()[:3]:
        env = DFAAgentEnvironment()
        obs = env.reset(
            split="train",
            scenario_id=card["scenario_id"],
            simulator_backend="mock",
            seed=args.seed,
            mode="demo",
        )
        while not obs.done:
            obs = env.step(run_baseline("default_policy", obs))
        trace = EpisodeTrace(**env.state.final_summary["trace"])
        export_trace_json(f"{args.output_dir}/{card['scenario_id']}.json", trace)
    print(f"Created sample rollouts in {args.output_dir}")


if __name__ == "__main__":
    main()
