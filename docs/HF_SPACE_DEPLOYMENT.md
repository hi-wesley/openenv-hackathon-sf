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

- `OPENAI_API_KEY` or `SIMULATOR_API_KEY`
- `SIMULATOR_MODEL`
- `SIMULATOR_BASE_URL` if using a non-OpenAI backend

5. Optional runtime variables:

- `DFA_AGENT_SCORER=noop`
- `DFA_AGENT_REVEAL_PERSONA=true`
- `DFA_AGENT_STRICT_BACKEND_ERRORS=false`

6. Wait for the Docker build to complete.
7. Open the Space URL and confirm:

- the app loads
- `/web` loads
- the default Playground tab exists
- the DFA Agent custom tab exists

## Recommended Secrets

- `OPENAI_API_KEY`: if the simulator should call OpenAI directly
- `SIMULATOR_API_KEY`: if pointing at another OpenAI-compatible backend
- `SIMULATOR_MODEL`: for example `gpt-4o-mini`, `Qwen/Qwen2.5-7B-Instruct`, or your hosted simulator name
- `SIMULATOR_BASE_URL`: example `https://api.openai.com/v1`

## Smoke Checklist After Deploy

1. Reset a mock episode.
2. Step a baseline once.
3. Switch to `openai_compatible`.
4. Reset the same scenario again.
5. Confirm the trace can be downloaded.

## Failure Modes

- Missing API key: the real simulator backend will terminate with a structured backend error
- Unsupported model output: inspect the simulator trace JSON in the custom tab
- No `/web`: confirm the Space is exposing port `7860` and the Docker command is the one from [server/Dockerfile](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/Dockerfile)

