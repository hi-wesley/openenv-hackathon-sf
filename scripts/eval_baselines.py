from __future__ import annotations

import argparse
import json
from pathlib import Path

from dfa_agent_env.baselines import BASELINE_REGISTRY, run_baseline
from dfa_agent_env.models import EpisodeTrace, EvalSummaryRow
from dfa_agent_env.scenario_schema import iter_scenarios
from dfa_agent_env.server.environment import DFAAgentEnvironment
from dfa_agent_env.server.trace import export_eval_summary_csv, export_traces_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DFA Agent baselines.")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--policy", default="all")
    parser.add_argument("--simulator-backend", default="mock")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--output-prefix", default="outputs/evals/baselines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = list(BASELINE_REGISTRY) if args.policy == "all" else [args.policy]
    rows = []
    traces = []
    for policy_name in policies:
        for scenario in iter_scenarios(split=args.split):
            env = DFAAgentEnvironment()
            observation = env.reset(
                split=args.split,
                scenario_id=scenario.scenario_id,
                simulator_backend=args.simulator_backend,
                seed=args.seed,
                max_turns=args.max_turns,
                mode="train",
                reveal_persona_after_done=True,
            )
            while not observation.done:
                observation = env.step(run_baseline(policy_name, observation))
            trace = EpisodeTrace(**env.state.final_summary["trace"])
            traces.append(trace)
            rows.append(
                EvalSummaryRow(
                    run_name="baseline_eval",
                    split=args.split,
                    scenario_id=scenario.scenario_id,
                    family=scenario.family,
                    simulator_backend=args.simulator_backend,
                    policy_name=policy_name,
                    total_reward=float(env.state.final_summary["total_reward"]),
                    shaped_reward=float(env.state.final_summary["shaped_reward_only"]),
                    final_satisfaction_reward=float(env.state.final_summary["final_satisfaction_reward"]),
                    task_completion=float(env.state.final_summary["task_completion_flag"]),
                    parse_validity=1.0,
                    turns_used=int(env.state.final_summary["turns_used"]),
                    invalid_action_count=env.state.invalid_action_count,
                    persona_summary=env.state.persona.reveal_dict(),
                    metadata={"done_reason": env.state.done_reason},
                )
            )
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    export_eval_summary_csv(prefix.with_suffix(".csv"), rows)
    export_traces_jsonl(prefix.with_suffix(".jsonl"), traces)
    prefix.with_suffix(".json").write_text(json.dumps([row.model_dump() for row in rows], indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} eval rows")


if __name__ == "__main__":
    main()

