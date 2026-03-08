# Environment Spec

## Name

- Product name: `DFA Agent`
- Package name: `dfa_agent_env`
- OpenEnv entrypoint: [server/app.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/app.py)

## Action

The assistant must emit strict JSON matching [AssistantAction](/Users/wesley/Documents/Projects/openenv-hackathon-sf/models.py):

```json
{
  "verbosity": "low|medium|high",
  "warmth": "low|medium|high",
  "humor": "low|medium|high",
  "formality": "low|medium|high",
  "directness": "low|medium|high",
  "initiative": "low|medium|high",
  "explanation_depth": "low|medium|high",
  "acknowledgement_style": "none|brief|empathetic",
  "message": "..."
}
```

Validation:

- `message` must be non-empty
- `message` must fit the configured character budget
- enum values are normalized and validated strictly
- invalid actions incur a penalty and can terminate the episode after the threshold is hit

## Observation

Visible observation fields:

- `scenario_id`
- `family`
- `turn_index`
- `max_turns`
- structured `conversation`
- `latest_user_message`
- `visible_context`
- `assistant_last_action_summary`
- `task_progress_visible`
- `done_reason`
- `available_style_axes`
- `episode_metrics_visible`
- `parse_error`
- `simulator_backend`
- `prompt_text`

The latent persona is hidden until the episode ends and reveal is enabled.

## State

State is serializable and includes:

- scenario and persona objects
- hidden user state
- conversation history
- per-turn logs
- reward components
- simulator trace
- scorer inputs and final summary

## Reset

Supported kwargs:

- `split`
- `scenario_id`
- `max_turns`
- `seed`
- `mode`
- `simulator_backend`
- `difficulty`
- `reveal_persona_after_done`
- `use_debug_heuristic_scorer`
- `family`

## Step

`step(action)`:

1. validates and normalizes the action
2. appends the assistant turn
3. applies deterministic hidden-state updates
4. calls the simulator backend
5. appends the user turn
6. computes reward components
7. checks termination
8. finalizes scorer inputs and trace when done

## Termination Conditions

- `max_turns_reached`
- `objective_achieved`
- `simulator_stopped`
- `invalid_action_threshold_reached`
- `fatal_backend_error`

