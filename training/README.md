# Training

The `training/` directory contains a minimal TRL GRPO setup for multi-turn rollouts against DFA Agent.

Key files:

- `colab_train_grpo.py`: Colab-friendly GRPO entrypoint with `use_vllm=True` and `vllm_mode="colocate"` by default.
- `colab_eval.py`: deterministic evaluation for a baseline, local HF checkpoint, or OpenAI-compatible endpoint.
- `build_dataset.py`: prompt-only dataset builder from the scenario splits.
- `rollout.py`: reusable multi-turn rollout helpers.
- `reward_adapter.py`: adapters that expose `env_reward` and metrics to TRL reward functions.
- `prompting.py`: prompt helpers and scenario-id extraction.

Recommended flow:

1. Start the environment locally or deploy it to a Hugging Face Space.
2. Build or load the prompt dataset.
3. Run `colab_train_grpo.py --fast-mode` first.
4. Save the adapter or checkpoint.
5. Run `colab_eval.py` on `val`, then `test`.

