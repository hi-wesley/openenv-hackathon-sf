# Architecture

## High-Level Flow

1. `reset(...)` selects a scenario from the JSONL split, samples a hidden persona, initializes hidden user state, and returns the first observation.
2. The assistant policy receives the visible observation and emits strict JSON for the action vector plus `message`.
3. `step(action)` validates and normalizes the action, appends the assistant turn, updates hidden state deterministically, and calls the selected simulator backend.
4. The simulator returns the next user message, latent state deltas, visible progress hints, and proxy signals.
5. The reward pipeline computes shaped reward components.
6. On termination, the environment constructs scorer inputs and calls the configured scorer if available.
7. The server stores a serializable trace for export, evaluation, and demo comparison.

## Modules

- [models.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/models.py): shared typed models for action, observation, state, simulator I/O, traces, and eval rows
- [serialization.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/serialization.py): prompt building, strict JSON extraction, parsing, and export helpers
- [baselines.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/baselines.py): deterministic baseline policies
- [scoring.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/scoring.py): scorer protocol and stub implementations
- [scenario_schema.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/scenario_schema.py): scenario loading and selection
- [server/environment.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/environment.py): main OpenEnv environment
- [server/persona_sampler.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/persona_sampler.py): persona sampling and deterministic hidden-state transitions
- [server/simulator_mock.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/simulator_mock.py): deterministic simulator
- [server/simulator_openai_compatible.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/simulator_openai_compatible.py): real simulator backend
- [server/web_ui.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/web_ui.py): custom Gradio dashboard tab
- [training/rollout.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/training/rollout.py): reusable multi-turn rollout helpers for GRPO and evaluation

## Modes

- `demo`: 3-5 turn episodes, rich prompt text, polished UI, persona reveal
- `train`: deterministic, configurable `max_turns`, stable reward fields
- `real`: same environment contract but with `openai_compatible` simulator for higher-fidelity evaluation or demoing

## Deployment Model

- Local dev: run the FastAPI/OpenEnv app directly
- Docker: build with [server/Dockerfile](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/Dockerfile)
- Hugging Face Space: deploy as a Docker Space with environment secrets
- Colab training: connect to the deployed Space or instantiate the environment locally from Python

## Scorer Plug-In Points

- Config chooses scorer name
- Environment builds scorer at startup/reset
- End-of-episode finalization calls `score(trace)`
- `state.scorer_inputs` stores everything a custom scorer needs

