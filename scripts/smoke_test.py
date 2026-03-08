from __future__ import annotations

import argparse
import json

from dfa_agent_env.baselines import default_policy
from dfa_agent_env.client import DFAAgentEnv
from dfa_agent_env.server.environment import DFAAgentEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick DFA Agent smoke test.")
    parser.add_argument("--base-url", default="")
    args = parser.parse_args()

    client = (
        DFAAgentEnv(base_url=args.base_url).sync()
        if args.base_url
        else DFAAgentEnv(environment=DFAAgentEnvironment()).sync()
    )
    result = client.reset(split="train", seed=3, mode="train", simulator_backend="mock")
    obs = result.observation
    action = default_policy(obs)
    result = client.step(action)
    state = client.state()
    payload = {
        "scenario_id": obs.scenario_id,
        "next_turn": result.observation.turn_index,
        "reward": result.reward,
        "done": result.done,
        "simulator_backend": state.simulator_backend,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

