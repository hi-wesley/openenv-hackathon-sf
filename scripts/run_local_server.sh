#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

python -m dfa_agent_env.server.app --host 0.0.0.0 --port 7860
