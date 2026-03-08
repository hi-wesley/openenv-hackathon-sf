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

from dfa_agent_env.baselines import default_policy
from dfa_agent_env.models import AssistantAction, EpisodeTrace
from dfa_agent_env.server.environment import DFAAgentEnvironment


class EnvironmentTests(unittest.TestCase):
    def test_multi_turn_rollout_finishes(self) -> None:
        env = DFAAgentEnvironment()
        obs = env.reset(
            split="train",
            scenario_id="email_boss_deadline_train_01",
            seed=7,
            max_turns=4,
            mode="train",
            simulator_backend="mock",
            reveal_persona_after_done=True,
        )
        while not obs.done:
            obs = env.step(default_policy(obs))
        self.assertIn(env.state.done_reason, {"objective_achieved", "simulator_stopped", "max_turns_reached"})
        trace = EpisodeTrace(**env.state.final_summary["trace"])
        self.assertEqual(len(trace.turn_logs), env.state.turn_index)
        self.assertEqual(len(trace.reward_components), env.state.turn_index)

    def test_invalid_action_threshold(self) -> None:
        env = DFAAgentEnvironment()
        obs = env.reset(split="train", seed=9, max_turns=4, mode="train", simulator_backend="mock")
        bad_action = AssistantAction(message="")
        obs = env.step(bad_action)
        self.assertFalse(obs.done)
        obs = env.step(bad_action)
        self.assertTrue(obs.done)
        self.assertEqual(env.state.done_reason, "invalid_action_threshold_reached")


if __name__ == "__main__":
    unittest.main()
