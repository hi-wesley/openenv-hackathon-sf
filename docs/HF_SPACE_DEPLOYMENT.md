# Hugging Face Space Deployment

## Target

Deploy this repo as a **Docker Space**.

## Files Used

- [server/Dockerfile](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/Dockerfile)
- [server/requirements.txt](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/requirements.txt)
- [server/app.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/app.py)
- [openenv.yaml](/Users/wesley/Documents/Projects/openenv-hackathon-sf/openenv.yaml)

## Steps

1. Create a new Space on Hugging Face.
2. Choose **Docker** as the SDK.
3. Push this repository to the Space.
4. In the Space settings, add secrets:

- `SIMULATOR_API_KEY` if using an external OpenAI-compatible backend

5. Optional runtime variables:

- `DFA_AGENT_SCORER=noop`
- `DFA_AGENT_REVEAL_PERSONA=true`
- `DFA_AGENT_STRICT_BACKEND_ERRORS=false`

6. Wait for the Docker build to complete.
7. Open the Space URL and confirm:

- the app loads
- `/web` loads
- the default Playground tab exists
- `/dashboard` loads

## Recommended Secrets

- `SIMULATOR_API_KEY`: if pointing at another OpenAI-compatible backend
- `SIMULATOR_MODEL`: if you want to override the default local model path
- `SIMULATOR_BASE_URL`: only if you want the optional `openai_compatible` backend

## Smoke Checklist After Deploy

1. Reset a mock episode.
2. Step a baseline once.
3. Switch to `local_hf`.
4. Reset the same scenario again.
5. Confirm the trace can be downloaded from `/dashboard`.

## Failure Modes

- Missing local model/runtime deps: the `local_hf` simulator backend will terminate with a structured backend error
- Unsupported model output: inspect the simulator trace JSON in `/dashboard`
- No `/web`: confirm the Space is exposing port `7860` and the Docker command is the one from [server/Dockerfile](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/Dockerfile)
