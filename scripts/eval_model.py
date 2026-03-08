from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from dfa_agent_env.baselines import BASELINE_REGISTRY, run_baseline
from dfa_agent_env.models import AssistantAction, EpisodeTrace, EvalSummaryRow
from dfa_agent_env.scenario_schema import iter_scenarios
from dfa_agent_env.serialization import parse_action_response
from dfa_agent_env.server.environment import DFAAgentEnvironment
from dfa_agent_env.server.trace import export_eval_summary_csv, export_traces_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained or remote model on DFA Agent.")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--policy", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--api-base-url", default="")
    parser.add_argument("--api-model", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--simulator-backend", default="local_hf")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--output-prefix", default="outputs/evals/model_eval")
    return parser.parse_args()


def _make_generator(args: argparse.Namespace):
    if args.policy:
        if args.policy not in BASELINE_REGISTRY:
            raise ValueError(f"Unknown baseline policy {args.policy}")
        return lambda obs: run_baseline(args.policy, obs), args.policy
    if args.model_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto")

        def _generate(obs):
            inputs = tokenizer(obs.prompt_text, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            parsed = parse_action_response(text, allow_message_only=True)
            return parsed.action or AssistantAction.default(text.strip() or "I can help.")

        return _generate, Path(args.model_path).name
    if args.api_base_url:
        def _generate(obs):
            body = {
                "model": args.api_model,
                "messages": [{"role": "user", "content": obs.prompt_text}],
                "temperature": 0.0,
            }
            request = urllib.request.Request(
                url=args.api_base_url.rstrip("/") + "/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {args.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"]
            parsed = parse_action_response(text, allow_message_only=True)
            return parsed.action or AssistantAction.default(text.strip() or "I can help.")

        return _generate, args.api_model or "remote-model"
    raise ValueError("Choose one of --policy, --model-path, or --api-base-url.")


def main() -> None:
    args = parse_args()
    generator, run_name = _make_generator(args)
    rows = []
    traces = []
    for scenario in iter_scenarios(split=args.split):
        env = DFAAgentEnvironment()
        obs = env.reset(
            split=args.split,
            scenario_id=scenario.scenario_id,
            simulator_backend=args.simulator_backend,
            seed=args.seed,
            max_turns=args.max_turns,
            mode="train",
        )
        while not obs.done:
            obs = env.step(generator(obs))
        trace = EpisodeTrace(**env.state.final_summary["trace"])
        traces.append(trace)
        rows.append(
            EvalSummaryRow(
                run_name=run_name,
                split=args.split,
                scenario_id=scenario.scenario_id,
                family=scenario.family,
                simulator_backend=args.simulator_backend,
                policy_name=run_name,
                total_reward=float(env.state.final_summary["total_reward"]),
                shaped_reward=float(env.state.final_summary["shaped_reward_only"]),
                final_satisfaction_reward=float(env.state.final_summary["final_satisfaction_reward"]),
                task_completion=float(env.state.final_summary["task_completion_flag"]),
                parse_validity=1.0,
                turns_used=int(env.state.final_summary["turns_used"]),
                invalid_action_count=env.state.invalid_action_count,
                customer_summary=env.state.final_summary.get("customer_summary", {}),
                metadata={"done_reason": env.state.done_reason},
            )
        )
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    export_eval_summary_csv(prefix.with_suffix(".csv"), rows)
    export_traces_jsonl(prefix.with_suffix(".jsonl"), traces)
    prefix.with_suffix(".json").write_text(json.dumps([row.model_dump() for row in rows], indent=2), encoding="utf-8")
    print(f"Wrote evaluation artifacts with prefix {prefix}")


if __name__ == "__main__":
    main()
