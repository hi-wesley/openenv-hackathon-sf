# Demo Script

## 3-5 Minute Judge Demo

1. Open the DFA Agent Space and show that the default OpenEnv Playground still exists under `/web`.
2. Open `/dashboard`.
3. Pick one of the story cards: `Late Delivery Refund`, `Damaged Item Replacement`, or `Surprise Billing Cancellation`.
4. Show the control panel: scenario, simulator backend, max turns, seed, assistant backend, and baseline policy.
5. Reset the episode and explain that the customer opening message is produced by a simulator model, not typed manually.
6. Click `Step Baseline` or use the local assistant backend and narrate that both sides can now be driven by the same local Qwen model.
7. Highlight the live customer emotion scores: happiness, anger, annoyance, gratitude.
8. End the episode and point out the final customer emotion scores and satisfaction score.
9. Run `Compare Baselines` to show different fixed response styles on the same case.
10. Open the trace comparator and mention that before/after training traces can be loaded side by side.
11. Close by explaining that the final scorer is still pluggable, so a stronger custom judge can be added later without changing the environment contract.

## Narrative Framing

- Primary: multi-agent interaction between assistant and simulated customer
- Secondary: personalized task completion under repeated customer feedback
- Hook: the assistant is not just generating polite text, it is trying to change measurable customer outcomes over multiple turns
