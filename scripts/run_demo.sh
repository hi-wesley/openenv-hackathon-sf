#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

export DFA_AGENT_DEFAULT_MODE=demo
export DFA_AGENT_SIMULATOR_BACKEND="${DFA_AGENT_SIMULATOR_BACKEND:-local_hf}"
python -m dfa_agent_env.server.app --host 0.0.0.0 --port 7860
