from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from .models import EpisodeTrace, SatisfactionScoreResult


class BaseSatisfactionScorer(ABC):
    name = "base"

    @abstractmethod
    def score(self, trace: EpisodeTrace) -> SatisfactionScoreResult:
        raise NotImplementedError


class NoopSatisfactionScorer(BaseSatisfactionScorer):
    name = "noop"

    def score(self, trace: EpisodeTrace) -> SatisfactionScoreResult:
        return SatisfactionScoreResult(
            score=None,
            available=False,
            reason="No final satisfaction scorer configured.",
            metadata={"scorer": self.name},
        )


class ConstantSatisfactionScorer(BaseSatisfactionScorer):
    name = "constant"

    def __init__(self, constant: float = 0.25) -> None:
        self.constant = float(constant)

    def score(self, trace: EpisodeTrace) -> SatisfactionScoreResult:
        return SatisfactionScoreResult(
            score=self.constant,
            available=True,
            reason="Constant development scorer.",
            metadata={"scorer": self.name, "constant": self.constant},
        )


class DebugProxySatisfactionScorer(BaseSatisfactionScorer):
    name = "debug_proxy"

    def score(self, trace: EpisodeTrace) -> SatisfactionScoreResult:
        if not trace.turn_logs:
            return SatisfactionScoreResult(
                score=0.0,
                available=True,
                reason="No turns available for debug proxy.",
                metadata={"scorer": self.name},
            )
        alignment = sum(log.reward_components.alignment_proxy for log in trace.turn_logs) / len(trace.turn_logs)
        progress = float(trace.hidden_state_final.goal_progress)
        trust = float(trace.hidden_state_final.trust)
        value = max(0.0, min(1.0, 0.4 * alignment + 0.4 * progress + 0.2 * trust))
        return SatisfactionScoreResult(
            score=value,
            available=True,
            reason="Debug-only proxy scorer derived from shaped signals.",
            metadata={"scorer": self.name, "alignment": alignment, "progress": progress, "trust": trust},
        )


SCORER_REGISTRY: Dict[str, type[BaseSatisfactionScorer]] = {
    NoopSatisfactionScorer.name: NoopSatisfactionScorer,
    ConstantSatisfactionScorer.name: ConstantSatisfactionScorer,
    DebugProxySatisfactionScorer.name: DebugProxySatisfactionScorer,
}


def build_scorer(name: str, constant: float = 0.25) -> BaseSatisfactionScorer:
    normalized = (name or "noop").strip().lower()
    if normalized == ConstantSatisfactionScorer.name:
        return ConstantSatisfactionScorer(constant=constant)
    scorer_cls = SCORER_REGISTRY.get(normalized, NoopSatisfactionScorer)
    return scorer_cls()

