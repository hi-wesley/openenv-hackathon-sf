# Training And Eval

This document is the operator runbook for local dev, Docker checks, Hugging Face Spaces deployment, Colab GRPO training, and evaluation.

## A. Local Development And Sanity Check

1. Create a Python 3.11 environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

2. Install the package in editable mode with dev and train extras.

```bash
pip install -e ".[dev,train]"
```

3. Run the local server.

```bash
bash scripts/run_local_server.sh
```

4. In another shell, run the smoke test.

```bash
python scripts/smoke_test.py
```

5. Open the local web UI.

- Visit `http://localhost:7860/web`
- Confirm the default Playground tab loads
- Confirm the DFA Agent custom dashboard tab loads

6. Test one mock episode and one baseline.

- Choose `mock` as the simulator backend
- Reset a scenario from the custom tab
- Use `default_policy`
- Step through until the episode ends

## B. Docker Sanity Check

1. Build the Docker image.

```bash
docker build -t dfa-agent -f server/Dockerfile .
```

2. Run the container locally.

```bash
docker run --rm -p 7860:7860 dfa-agent
```

3. Confirm the health endpoint works.

- Visit `http://localhost:7860/health`

4. Confirm `/web` works.

- Visit `http://localhost:7860/web`

5. Confirm trace export works.

- Complete an episode in the custom tab
- Verify a JSON trace appears in `outputs/logs/`

## C. Hugging Face Space Deployment

1. Create a new **Docker Space** on Hugging Face.
2. Add the required secrets and variables.

- `OPENAI_API_KEY` or `SIMULATOR_API_KEY`
- `SIMULATOR_MODEL`
- `SIMULATOR_BASE_URL` if you are not using OpenAI
- optional `DFA_AGENT_SCORER`, `DFA_AGENT_REVEAL_PERSONA`

3. Push this repo to the Space.
4. Confirm the app is serving.
5. Confirm the custom web tab renders.
6. Confirm the simulator backend works if configured.

See [HF_SPACE_DEPLOYMENT.md](/Users/wesley/Documents/Projects/openenv-hackathon-sf/docs/HF_SPACE_DEPLOYMENT.md) for the exact checklist.

## D. Colab Training

1. Open [notebooks/adaptive_persona_env_minimal_colab.ipynb](/Users/wesley/Documents/Projects/openenv-hackathon-sf/notebooks/adaptive_persona_env_minimal_colab.ipynb) in Colab.
2. Install TRL, OpenEnv core, and this environment package.

Example Colab install cell:

```bash
pip install trl transformers datasets accelerate peft openenv-core
pip install git+https://github.com/<your-account>/<your-space-repo>.git
```

3. Point the client at the deployed HF Space or run locally in the notebook.

- Space URL example: `https://your-hf-space-url.hf.space`

4. Use fast mode first.

```bash
python training/colab_train_grpo.py \
  --env-url https://your-hf-space-url.hf.space \
  --fast-mode \
  --use-vllm \
  --vllm-mode colocate
```

5. Run a very short job.

- start with `--max-steps 8`
- keep `--max-turns 3`
- use the mock simulator first

6. Save the checkpoint or LoRA adapter from the output directory.
7. Plot reward curves from the trainer logs or exported summaries.

## E. Baseline Evaluation

1. Run the baseline eval on the validation split.

```bash
python scripts/eval_baselines.py --split val --policy all --simulator-backend mock
```

2. Export CSV and JSONL summaries.

- `outputs/evals/baselines.csv`
- `outputs/evals/baselines.jsonl`
- `outputs/evals/baselines.json`

3. Save representative traces.

```bash
python scripts/export_demo_traces.py
```

## F. Trained-Model Evaluation

1. Load a trained adapter/checkpoint or point at an OpenAI-compatible endpoint.

Local model example:

```bash
python scripts/eval_model.py --split val --model-path /path/to/checkpoint
```

Remote endpoint example:

```bash
python scripts/eval_model.py \
  --split test \
  --api-base-url https://your-endpoint/v1 \
  --api-model your-model-name \
  --api-key $OPENAI_API_KEY
```

2. Compare against baselines.
3. Export reward summaries and traces.
4. Generate before/after examples for the custom dashboard comparator.

## Notes

- The final satisfaction scorer is intentionally stubbed by default.
- For training, shaped rewards are available immediately.
- For higher-fidelity demos or evaluation, switch to `openai_compatible` simulator mode.

