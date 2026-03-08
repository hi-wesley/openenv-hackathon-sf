# Architecture

## High-Level Flow

1. `reset(...)` selects one of the customer-service scenarios, initializes the chat state, and requests the opening customer message from the configured simulator.
2. The assistant policy receives the visible observation and emits strict JSON with only `message`.
3. `step(action)` validates the action, applies deterministic hidden-state updates, and calls the simulator for the next customer reply.
4. The customer reply is scored on `happiness`, `anger`, `annoyance`, and `gratitude`.
5. The reward pipeline converts emotion movement, resolution progress, and format validity into shaped reward.
6. When the episode ends, the environment builds a full trace and calls the configured final scorer.
7. The server exposes both the default OpenEnv UI at `/web` and the customer-service dashboard at `/dashboard`.

## Modules

- [models.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/models.py): action, observation, chat state, traces, and eval row models
- [serialization.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/serialization.py): prompt building, JSON extraction, parsing, and export helpers
- [baselines.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/baselines.py): deterministic customer-service baselines
- [scoring.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/scoring.py): scorer protocol plus lightweight built-in scorers
- [scenario_schema.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/scenario_schema.py): scenario loading and selection
- [server/environment.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/environment.py): main OpenEnv environment
- [server/persona_sampler.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/persona_sampler.py): chat-state initialization and updates
- [server/reward_pipeline.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/reward_pipeline.py): shaped reward decomposition
- [server/simulator_mock.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/simulator_mock.py): deterministic offline customer simulator
- [server/simulator_local_hf.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/simulator_local_hf.py): local Qwen-driven customer simulator
- [server/simulator_openai_compatible.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/simulator_openai_compatible.py): optional external OpenAI-compatible simulator backend
- [server/web_ui.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/web_ui.py): Gradio dashboard for scenario selection, replay, and comparison
- [training/rollout.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/training/rollout.py): reusable multi-turn rollout loop for GRPO and eval

## Modes

- `demo`: short episodes and dashboard-friendly traces
- `train`: deterministic setup, stable reward fields, configurable `max_turns`
- `real`: same environment contract with the `local_hf` or optional `openai_compatible` customer simulator backend

## Deployment Model

- Local dev: run the FastAPI/OpenEnv app directly
- Docker: build with [server/Dockerfile](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/Dockerfile)
- Hugging Face Space: deploy as a Docker Space with API-key secrets
- Colab training: connect to the deployed Space or instantiate the environment locally from Python

## Scorer Plug-In Points

- Config chooses the scorer name
- Environment builds the scorer at startup and reset
- End-of-episode finalization calls `score(trace)`
- `state.scorer_inputs` stores the final scenario, conversation, and logs
