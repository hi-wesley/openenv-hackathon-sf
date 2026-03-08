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

from dfa_agent_env.scenario_schema import load_scenarios


class ScenarioDataTests(unittest.TestCase):
    def test_split_counts(self) -> None:
        self.assertEqual(len(load_scenarios("train")), 3)
        self.assertEqual(len(load_scenarios("val")), 3)
        self.assertEqual(len(load_scenarios("test")), 3)

    def test_family_distribution(self) -> None:
        train = load_scenarios("train")
        families = {}
        for scenario in train:
            families[scenario.family] = families.get(scenario.family, 0) + 1
        self.assertEqual(len(families), 3)
        self.assertTrue(all(count == 1 for count in families.values()))


if __name__ == "__main__":
    unittest.main()
