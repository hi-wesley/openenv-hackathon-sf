from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync server/requirements.txt from pyproject.toml.")
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--output", default="server/requirements.txt")
    args = parser.parse_args()

    with open(args.pyproject, "rb") as handle:
        data = tomllib.load(handle)
    deps = data.get("project", {}).get("dependencies", [])
    output_path = Path(args.output)
    output_path.write_text("\n".join(deps) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
