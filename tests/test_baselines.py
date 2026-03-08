from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "dfa_agent_env" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "dfa_agent_env",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["dfa_agent_env"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

from dfa_agent_env.baselines import BASELINE_REGISTRY, run_baseline
from dfa_agent_env.server.environment import DFAAgentEnvironment


class BaselineTests(unittest.TestCase):
    def test_all_baselines_return_valid_actions(self) -> None:
        env = DFAAgentEnvironment()
        obs = env.reset(split="train", seed=5, mode="train", simulator_backend="mock")
        for name in BASELINE_REGISTRY:
            action = run_baseline(name, obs)
            self.assertEqual(action.validate_action(700), [])


if __name__ == "__main__":
    unittest.main()
