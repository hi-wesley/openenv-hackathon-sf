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

from dfa_agent_env.models import (
    ChatState,
    ConversationMessage,
    EmotionScores,
    EpisodeTrace,
    SatisfactionScoreResult,
    ScenarioRecord,
)
from dfa_agent_env.scoring import EmotionBalanceScorer


class EmotionScorerTests(unittest.TestCase):
    def test_emotion_balance_scorer_returns_available_score(self) -> None:
        scorer = EmotionBalanceScorer()
        result = scorer.score(
            EpisodeTrace(
                scenario=ScenarioRecord(
                    scenario_id="s1",
                    split="train",
                    family="late_delivery_refund",
                    title="title",
                    visible_context="context",
                    initial_user_message="My order is late.",
                    task_success_criteria=["help"],
                    allowed_turns=4,
                    difficulty="easy",
                    simulator_instructions="stay in role",
                    tags=[],
                ),
                chat_state_final=ChatState(emotion_scores=EmotionScores(), satisfaction_score=0.0),
                conversation=[ConversationMessage(role="user", content="Thanks, that helps a lot.")],
                turn_logs=[],
                final_summary={},
                scorer_result=SatisfactionScoreResult(),
                reward_components=[],
                simulator_backend="mock",
                mode="demo",
            )
        )
        self.assertTrue(result.available)
        self.assertGreaterEqual(result.score, -1.0)
        self.assertLessEqual(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()
