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

from dfa_agent_env.serialization import extract_first_json_object, parse_action_response


class SerializationTests(unittest.TestCase):
    def test_extract_first_json_object(self) -> None:
        blob = extract_first_json_object('noise {"message":"hi","verbosity":"medium"} trailing')
        self.assertEqual(blob, '{"message":"hi","verbosity":"medium"}')

    def test_parse_action_response(self) -> None:
        text = """
        Sure:
        {"verbosity":"high","warmth":"medium","humor":"low","formality":"high","directness":"medium","initiative":"medium","explanation_depth":"high","acknowledgement_style":"brief","message":"Here is a polished draft."}
        """
        parsed = parse_action_response(text)
        self.assertIsNone(parsed.parse_error)
        self.assertEqual(parsed.action.message, "Here is a polished draft.")

    def test_message_only_fallback(self) -> None:
        parsed = parse_action_response("Just send a short apology and ask for a new time.", allow_message_only=True)
        self.assertTrue(parsed.used_message_fallback)
        self.assertIsNotNone(parsed.action)


if __name__ == "__main__":
    unittest.main()
