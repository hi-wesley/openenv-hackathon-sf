from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

from .config import DATA_DIR
from .models import ScenarioRecord


def _scenario_path(split: str) -> Path:
    return DATA_DIR / f"scenarios_{split}.jsonl"


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=8)
def load_scenarios(split: str | None = None) -> List[ScenarioRecord]:
    splits = [split] if split else ["train", "val", "test"]
    records: List[ScenarioRecord] = []
    for item in splits:
        for row in _load_jsonl(_scenario_path(item)):
            records.append(ScenarioRecord(**row))
    return records


@lru_cache(maxsize=1)
def load_demo_story_cards() -> list[dict]:
    path = DATA_DIR / "demo_story_cards.json"
    return json.loads(path.read_text(encoding="utf-8"))


def iter_scenarios(
    *,
    split: str | None = None,
    family: str | None = None,
    difficulty: str | None = None,
) -> Iterable[ScenarioRecord]:
    for scenario in load_scenarios(split):
        if family and scenario.family != family:
            continue
        if difficulty and scenario.difficulty != difficulty:
            continue
        yield scenario


def get_scenario(scenario_id: str) -> ScenarioRecord:
    for scenario in load_scenarios(None):
        if scenario.scenario_id == scenario_id:
            return scenario
    raise KeyError(f"Unknown scenario_id: {scenario_id}")


def select_scenario(
    *,
    split: str = "train",
    family: str | None = None,
    difficulty: str | None = None,
    scenario_id: str | None = None,
    seed: int | None = None,
) -> ScenarioRecord:
    if scenario_id:
        return get_scenario(scenario_id)
    candidates = list(iter_scenarios(split=split, family=family, difficulty=difficulty))
    if not candidates:
        raise ValueError(f"No scenarios found for split={split}, family={family}, difficulty={difficulty}")
    rng = random.Random(seed)
    return rng.choice(candidates)

