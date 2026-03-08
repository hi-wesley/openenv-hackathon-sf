from __future__ import annotations

import argparse
import json
from pathlib import Path

from dfa_agent_env.baselines import BASELINE_REGISTRY, run_baseline
from dfa_agent_env.models import AssistantAction, EpisodeTrace, EvalSummaryRow
from dfa_agent_env.scenario_schema import iter_scenarios
from dfa_agent_env.server.environment import DFAAgentEnvironment
from dfa_agent_env.server.trace import export_eval_summary_csv, export_trace_json, export_traces_jsonl
from dfa_agent_env.serialization import parse_action_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a baseline or model on DFA Agent.")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--policy", default="default_policy")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--api-base-url", default="")
    parser.add_argument("--api-model", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--simulator-backend", default="mock")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-prefix", default="outputs/evals/colab_eval")
    return parser.parse_args()


def _make_generator(args: argparse.Namespace):
    if args.policy in BASELINE_REGISTRY:
        return lambda obs: run_baseline(args.policy, obs)
    if args.model_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto")

        def _generate(obs):
            inputs = tokenizer(obs.prompt_text, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            parsed = parse_action_response(text, allow_message_only=True)
            return parsed.action or AssistantAction.default(text.strip() or "I can help with this.")

        return _generate
    if args.api_base_url:
        import urllib.request

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
            return parsed.action or AssistantAction.default(text.strip() or "I can help with this.")

        return _generate
    raise ValueError("Provide a baseline policy, --model-path, or --api-base-url.")


def main() -> None:  # pragma: no cover
    args = parse_args()
    generator = _make_generator(args)
    rows = []
    traces = []
    for scenario in iter_scenarios(split=args.split):
        env = DFAAgentEnvironment()
        observation = env.reset(
            split=args.split,
            scenario_id=scenario.scenario_id,
            max_turns=args.max_turns,
            seed=args.seed,
            mode="train",
            simulator_backend=args.simulator_backend,
            reveal_persona_after_done=True,
        )
        while not observation.done:
            observation = env.step(generator(observation))
        trace = env.state.final_summary["trace"]
        traces.append(trace)
        rows.append(
            EvalSummaryRow(
                run_name=args.policy or args.model_path or args.api_model,
                split=args.split,
                scenario_id=scenario.scenario_id,
                family=scenario.family,
                simulator_backend=args.simulator_backend,
                policy_name=args.policy if args.policy in BASELINE_REGISTRY else "model",
                total_reward=float(env.state.final_summary.get("total_reward", 0.0)),
                shaped_reward=float(env.state.final_summary.get("shaped_reward_only", 0.0)),
                final_satisfaction_reward=float(env.state.final_summary.get("final_satisfaction_reward", 0.0)),
                task_completion=float(env.state.final_summary.get("task_completion_flag", 0.0)),
                parse_validity=1.0,
                turns_used=int(env.state.final_summary.get("turns_used", 0)),
                invalid_action_count=int(env.state.invalid_action_count),
                persona_summary=env.state.persona.reveal_dict(),
                metadata={"done_reason": env.state.done_reason},
            )
        )
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    export_eval_summary_csv(prefix.with_suffix(".csv"), rows)
    export_traces_jsonl(prefix.with_suffix(".jsonl"), [EpisodeTrace(**trace) for trace in traces])
    prefix.with_suffix(".json").write_text(json.dumps([row.model_dump() for row in rows], indent=2), encoding="utf-8")
    print(f"Wrote evaluation outputs with prefix {prefix}")


if __name__ == "__main__":  # pragma: no cover
    main()
