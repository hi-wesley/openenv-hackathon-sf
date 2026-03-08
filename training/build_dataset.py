from __future__ import annotations

import argparse
import json
from pathlib import Path

from dfa_agent_env.scenario_schema import iter_scenarios
from dfa_agent_env.server.environment import DFAAgentEnvironment


def build_prompt_records(
    *,
    split: str,
    mode: str = "train",
    max_turns: int = 4,
    seed: int = 7,
) -> list[dict]:
    env = DFAAgentEnvironment()
    records = []
    for scenario in iter_scenarios(split=split):
        observation = env.reset(
            split=split,
            scenario_id=scenario.scenario_id,
            mode=mode,
            max_turns=max_turns,
            seed=seed,
            simulator_backend="mock",
        )
        records.append(
            {
                "prompt": observation.prompt_text,
                "scenario_id": scenario.scenario_id,
                "family": scenario.family,
                "split": scenario.split,
                "max_turns": max_turns,
                "mode": mode,
            }
        )
    return records


def main() -> None:  # pragma: no cover - simple CLI wrapper
    parser = argparse.ArgumentParser(description="Build prompt-only datasets for DFA Agent.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--mode", default="train")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/evals/prompt_dataset_train.jsonl"))
    args = parser.parse_args()

    rows = build_prompt_records(split=args.split, mode=args.mode, max_turns=args.max_turns, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()

