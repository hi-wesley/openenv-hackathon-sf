# Scorer Plugin Guide

The repository intentionally does **not** implement the final satisfaction scorer. The environment is already wired so you can drop your scorer in with minimal edits.

## Interfaces

See [scoring.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/scoring.py).

Required interface:

```python
class BaseSatisfactionScorer(ABC):
    def score(self, trace: EpisodeTrace) -> SatisfactionScoreResult:
        ...
```

Result object:

```python
class SatisfactionScoreResult(BaseModel):
    score: float | None
    available: bool
    reason: str | None
    metadata: dict
```

## What The Scorer Receives

The `EpisodeTrace` passed to the scorer includes:

- scenario metadata
- revealed persona object
- final hidden state
- full conversation
- per-turn logs
- reward components
- final summary
- simulator backend, mode, and seed

The raw scorer inputs are also stored in `state.scorer_inputs`.

## Where To Add Your Scorer

1. Implement a new scorer class in [scoring.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/scoring.py) or a separate module that imports the same models.
2. Register it in `SCORER_REGISTRY`.
3. Set `DFA_AGENT_SCORER=<your_name>` or change the default in [config.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/config.py).

## Environment Hook Points

- scorer creation: [server/environment.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/environment.py)
- scorer selection: [config.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/config.py)
- episode-end call: [server/environment.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/environment.py)

## Design Guidance

- Keep the scorer stateless where possible.
- Return `available=False` instead of raising for expected unavailable cases.
- Put extra judge diagnostics in `metadata`.
- Do not overload shaped reward logic into the final scorer path unless that is explicitly your intended design.

