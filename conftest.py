import importlib.util
import sys
from pathlib import Path

# Bootstrap the dfa_agent_env package before pytest tries to import
# __init__.py standalone (which fails due to relative imports).
# Pytest imports __init__.py as module name "__init__", so we must
# also alias it under that name to prevent a duplicate import.
ROOT = Path(__file__).resolve().parent
if "dfa_agent_env" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "dfa_agent_env",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["dfa_agent_env"] = module
    spec.loader.exec_module(module)

sys.modules.setdefault("__init__", sys.modules["dfa_agent_env"])
