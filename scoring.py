from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from .models import EmotionScores, EpisodeTrace, SatisfactionScoreResult
from .server.utils import score_customer_emotions


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


class EmotionBalanceScorer(BaseSatisfactionScorer):
    name = "emotion_balance"

    def score(self, trace: EpisodeTrace) -> SatisfactionScoreResult:
        final_customer_text = ""
        for message in reversed(trace.conversation):
            if message.role == "user":
                final_customer_text = message.content
                break
        if not final_customer_text:
            return SatisfactionScoreResult(
                score=None,
                available=False,
                reason="No final customer message available to score.",
                metadata={"scorer": self.name},
            )
        emotions = score_customer_emotions(final_customer_text)
        return SatisfactionScoreResult(
            score=emotions.composite(),
            available=True,
            reason="Final score computed from customer emotion balance.",
            metadata={
                "scorer": self.name,
                "emotion_scores": emotions.model_dump(),
                "final_customer_text": final_customer_text,
            },
        )


SCORER_REGISTRY: Dict[str, type[BaseSatisfactionScorer]] = {
    NoopSatisfactionScorer.name: NoopSatisfactionScorer,
    ConstantSatisfactionScorer.name: ConstantSatisfactionScorer,
    EmotionBalanceScorer.name: EmotionBalanceScorer,
}


def build_scorer(name: str, constant: float = 0.25) -> BaseSatisfactionScorer:
    normalized = (name or "emotion_balance").strip().lower()
    if normalized == ConstantSatisfactionScorer.name:
        return ConstantSatisfactionScorer(constant=constant)
    scorer_cls = SCORER_REGISTRY.get(normalized, NoopSatisfactionScorer)
    return scorer_cls()
