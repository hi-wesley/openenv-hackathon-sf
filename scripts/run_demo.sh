#!/usr/bin/env bash
set -euo pipefail

export DFA_AGENT_DEFAULT_MODE=demo
export DFA_AGENT_SIMULATOR_BACKEND="${DFA_AGENT_SIMULATOR_BACKEND:-mock}"
python -m dfa_agent_env.server.app --host 0.0.0.0 --port 7860

