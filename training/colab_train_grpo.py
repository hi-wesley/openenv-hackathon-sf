from __future__ import annotations

import argparse
import json
from pathlib import Path

from dfa_agent_env.training.build_dataset import build_prompt_records
from dfa_agent_env.training.reward_adapter import reward_from_env
from dfa_agent_env.training.rollout import make_env_adapter, run_grpo_rollout_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Colab-friendly GRPO trainer for DFA Agent.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--env-url", default="", help="Optional deployed Space URL.")
    parser.add_argument("--output-dir", default="outputs/checkpoints/dfa-agent-grpo")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--simulator-backend", default="local_hf")
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--use-vllm", action="store_true", default=True)
    parser.add_argument("--vllm-mode", default="colocate")
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--allow-message-only", action="store_true")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - runtime entrypoint
    args = parse_args()
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    rows = build_prompt_records(
        split=args.train_split,
        mode="train",
        max_turns=args.max_turns,
        seed=7,
        simulator_backend=args.simulator_backend,
    )
    if args.fast_mode:
        rows = rows[:8]
        args.max_steps = min(args.max_steps, 8)
        args.max_turns = min(args.max_turns, 3)
        args.num_generations = min(args.num_generations, 2)
    dataset = Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    env = make_env_adapter(args.env_url or None)

    def rollout_func(prompts, trainer):
        prompt_ids_batch = []
        completion_ids_batch = []
        logprobs_batch = []
        env_rewards = []
        task_completion = []
        parse_valid = []
        turns_used = []
        traces = []
        for prompt in prompts:
            rollout = run_grpo_rollout_episode(
                trainer=trainer,
                tokenizer=tokenizer,
                prompt_text=prompt,
                env=env,
                max_turns=args.max_turns,
                simulator_backend=args.simulator_backend,
                allow_message_only=args.allow_message_only,
            )
            prompt_ids_batch.append(rollout["prompt_ids"])
            completion_ids_batch.append(rollout["completion_ids"])
            logprobs_batch.append(rollout["logprobs"])
            env_rewards.append(rollout["env_reward"])
            task_completion.append(rollout["task_completion"])
            parse_valid.append(rollout["parse_valid"])
            turns_used.append(rollout["turns_used"])
            traces.append(rollout["trace_json"])
        return {
            "prompt_ids": prompt_ids_batch,
            "completion_ids": completion_ids_batch,
            "logprobs": logprobs_batch,
            "env_reward": env_rewards,
            "task_completion": task_completion,
            "parse_valid": parse_valid,
            "turns_used": turns_used,
            "trace_json": traces,
        }

    config = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=1,
        save_steps=max(1, args.max_steps // 2),
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=reward_from_env,
        train_dataset=dataset,
        args=config,
        rollout_func=rollout_func,
    )
    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_dir) / "run_config.json"
    summary_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Saved run config to {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
