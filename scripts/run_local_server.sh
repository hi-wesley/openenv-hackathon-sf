#!/usr/bin/env bash
set -euo pipefail

python -m dfa_agent_env.server.app --host 0.0.0.0 --port 7860

