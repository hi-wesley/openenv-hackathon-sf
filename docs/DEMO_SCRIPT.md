# Demo Script

## 3-5 Minute Judge Demo

1. Open the DFA Agent Space and show that the default OpenEnv Playground still exists under `/web`.
2. Switch to the custom DFA Agent dashboard tab.
3. Pick `Boss Email Under Deadline` or `Account Lockout With Panic` from the sample story cards.
4. Show the control panel: scenario family, scenario picker, simulator backend, max turns, seed, baseline policy, persona reveal.
5. Reset the episode and point out that the assistant only sees visible context and the conversation, not the hidden persona.
6. Click `Step Baseline` once or twice and narrate the strategy vector changing by turn.
7. Highlight the reward breakdown and strategy heatmap.
8. End the episode and reveal the hidden persona bars. Explain that the model is trying to infer this latent preference profile from dialogue evidence.
9. Run `Compare Baselines` to show concise vs warm-detailed vs formal-direct behavior on the same scenario.
10. Open the trace comparator and mention that before/after training traces can be loaded side by side.
11. Close by explaining that the final satisfaction scorer is pluggable, so the environment can be used now for demos and training while a stronger judge is added later.

## Narrative Framing

- Primary: multi-agent interaction between assistant and simulated user
- Secondary: personalized task completion under hidden preference uncertainty
- Hook: style alone is not enough; the assistant must infer what the user actually needs over multiple turns

