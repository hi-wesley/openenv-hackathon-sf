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
        blob = extract_first_json_object('noise {"message":"hi"} trailing')
        self.assertEqual(blob, '{"message":"hi"}')

    def test_parse_action_response(self) -> None:
        text = """
        Sure:
        {"message":"I can help resolve this and move toward a refund."}
        """
        parsed = parse_action_response(text)
        self.assertIsNone(parsed.parse_error)
        self.assertEqual(parsed.action.message, "I can help resolve this and move toward a refund.")

    def test_message_only_fallback(self) -> None:
        parsed = parse_action_response("Just send a short apology and ask for a new time.", allow_message_only=True)
        self.assertTrue(parsed.used_message_fallback)
        self.assertIsNotNone(parsed.action)


if __name__ == "__main__":
    unittest.main()
