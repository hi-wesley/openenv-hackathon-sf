from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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
from dfa_agent_env.models import AssistantAction, EpisodeTrace, SimulatorOutput
from dfa_agent_env.server.environment import DFAAgentEnvironment


class EnvironmentTests(unittest.TestCase):
    def test_multi_turn_rollout_finishes(self) -> None:
        env = DFAAgentEnvironment()
        obs = env.reset(
            split="train",
            scenario_id="late_delivery_refund_train_01",
            seed=7,
            max_turns=4,
            mode="train",
            simulator_backend="mock",
        )
        while not obs.done:
            obs = env.step(default_policy(obs))
        self.assertIn(env.state.done_reason, {"objective_achieved", "simulator_stopped", "max_turns_reached"})
        trace = EpisodeTrace(**env.state.final_summary["trace"])
        self.assertEqual(len(trace.turn_logs), env.state.turn_index)
        self.assertEqual(len(trace.reward_components), env.state.turn_index)
        self.assertIn("customer_satisfaction_score", env.state.final_summary)

    def test_invalid_action_threshold(self) -> None:
        env = DFAAgentEnvironment()
        obs = env.reset(split="train", seed=9, max_turns=4, mode="train", simulator_backend="mock")
        bad_action = AssistantAction(message="x" * 5000)
        obs = env.step(bad_action)
        self.assertFalse(obs.done)
        obs = env.step(bad_action)
        self.assertTrue(obs.done)
        self.assertEqual(env.state.done_reason, "invalid_action_threshold_reached")

    def test_opening_backend_error_is_surfaced_without_fallback(self) -> None:
        class FailingSimulator:
            def generate_opening_message(self, sim_input):
                return SimulatorOutput(
                    user_message="",
                    continue_episode=False,
                    backend_error="opening failure",
                    simulator_notes=["forced failure"],
                )

            def generate_reply(self, sim_input):
                raise AssertionError("generate_reply should not run when reset fails")

        with patch("dfa_agent_env.server.environment.build_simulator", return_value=FailingSimulator()):
            env = DFAAgentEnvironment()
            obs = env.reset(split="train", seed=1, max_turns=4, mode="train", simulator_backend="local_hf")
        self.assertTrue(obs.done)
        self.assertEqual(env.state.done_reason, "fatal_backend_error")
        self.assertEqual(env.state.final_summary.get("backend_error"), "opening failure")
        self.assertEqual(env.state.conversation, [])


if __name__ == "__main__":
    unittest.main()
