# DFA Agent

DFA Agent is a multi-turn OpenEnv environment for training and demoing assistants that must infer hidden user preferences over a conversation, choose a structured communication strategy vector, and produce the actual natural-language reply.

The project is framed primarily as **Statement 1: Multi-Agent Interactions** and secondarily as **Statement 3.2: Personalized Tasks**. The assistant policy interacts with a simulated user whose latent preferences and internal state evolve across turns. The final satisfaction scorer is intentionally **pluggable** and ships as a stub so you can add your own end-of-episode logic later.

## Why It Matters

- Multi-turn personalization is closer to real assistant behavior than one-shot style labels.
- The assistant must infer preferences from dialogue, not hidden metadata.
- The project is ready for local dev, Docker, Hugging Face Spaces, and Colab-based TRL GRPO experiments.

## Features

- OpenEnv `0.2.1` server/client integration
- Multi-turn environment with 3-5 turn demo episodes and configurable training episodes
- Hidden persona sampler plus evolving hidden state
- Deterministic mock simulator and OpenAI-compatible simulator backend
- Strict structured assistant action format
- Shaped reward decomposition with a pluggable final scorer hook
- Polished custom Gradio dashboard layered on top of the default OpenEnv `/web` Playground
- Baselines, trace export, comparison widgets, training helpers, and eval scripts

## Repo Layout

Core package modules live at the repo root and are exposed as the `dfa_agent_env` package. The runtime environment server is under [server/app.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/app.py), training helpers are under [training/](/Users/wesley/Documents/Projects/openenv-hackathon-sf/training), scenario data is under [data/](/Users/wesley/Documents/Projects/openenv-hackathon-sf/data), and the main operator docs are under [docs/](/Users/wesley/Documents/Projects/openenv-hackathon-sf/docs).

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,train]"
python -m dfa_agent_env.server.app --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860/web`.

## Local Development

1. Install the package in editable mode.
2. Run `scripts/run_local_server.sh`.
3. Run `python scripts/smoke_test.py`.
4. Visit `/web` for the default Playground plus the custom DFA Agent dashboard tab.

## Docker

```bash
docker build -t dfa-agent -f server/Dockerfile .
docker run --rm -p 7860:7860 dfa-agent
```

## Hugging Face Spaces

This repo is structured for a **Docker Space**. See [docs/HF_SPACE_DEPLOYMENT.md](/Users/wesley/Documents/Projects/openenv-hackathon-sf/docs/HF_SPACE_DEPLOYMENT.md) for exact steps and required secrets.

## Training Overview

- Build prompt records with [training/build_dataset.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/training/build_dataset.py)
- Run fast GRPO training first with [training/colab_train_grpo.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/training/colab_train_grpo.py)
- Evaluate with [training/colab_eval.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/training/colab_eval.py) or [scripts/eval_model.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/scripts/eval_model.py)
- Compare against built-in baselines with [scripts/eval_baselines.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/scripts/eval_baselines.py)

See [docs/TRAINING_AND_EVAL.md](/Users/wesley/Documents/Projects/openenv-hackathon-sf/docs/TRAINING_AND_EVAL.md) for step-by-step operator instructions.

## Demo UI

The custom tab includes:

- transcript view
- control panel for scenario, seed, simulator backend, and baseline
- per-turn strategy heatmap
- reward breakdown table
- persona reveal bars
- baseline comparison widget
- trace download and side-by-side trace comparison
- sample story cards for demo day

## Baselines And Eval

Built-in baselines:

- `default_policy`
- `always_concise_policy`
- `always_warm_detailed_policy`
- `always_formal_direct_policy`

Evaluation outputs include JSON traces, JSONL batches, and CSV summaries with reward, completion, parse validity, turns used, and persona metadata.

## Screenshots / Demo Assets

Place screenshots, GIFs, or exported comparison traces in [outputs/](/Users/wesley/Documents/Projects/openenv-hackathon-sf/outputs) or link them from the docs. The custom UI already supports downloadable traces and side-by-side before/after comparison panels.

## Final Satisfaction Scorer

The final scorer is intentionally **not implemented**. The project ships with:

- `BaseSatisfactionScorer`
- `NoopSatisfactionScorer`
- `ConstantSatisfactionScorer`
- an optional `DebugProxySatisfactionScorer` for development-only sanity checks

To add your own scorer, follow [docs/SCORER_PLUGIN_GUIDE.md](/Users/wesley/Documents/Projects/openenv-hackathon-sf/docs/SCORER_PLUGIN_GUIDE.md).

