# Environment Spec

## Name

- Product name: `DFA Agent`
- Package name: `dfa_agent_env`
- OpenEnv entrypoint: [server/app.py](/Users/wesley/Documents/Projects/openenv-hackathon-sf/server/app.py)

## Scenario Families

The environment currently exposes three customer-service scenario families:

- `late_delivery_refund`
- `damaged_item_replacement`
- `surprise_billing_cancellation`

Each scenario contains visible context, a concrete support problem, success criteria, and simulator instructions.

## Action

The assistant must emit strict JSON matching [AssistantAction](/Users/wesley/Documents/Projects/openenv-hackathon-sf/models.py):

```json
{
  "message": "..."
}
```

Validation:

- `message` must be non-empty
- `message` must fit the configured character budget
- enum values are normalized and validated strictly
- invalid actions incur a penalty and can terminate the episode after the threshold is hit

## Customer Emotions

Every customer reply is scored on four `0..1` dimensions:

- `happiness`
- `anger`
- `annoyance`
- `gratitude`

These scores are visible in the observation and are also logged in the turn trace.

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
- `customer_emotion_scores`
- `customer_satisfaction_score`
- `done_reason`
- `episode_metrics_visible`
- `parse_error`
- `simulator_backend`
- `prompt_text`

## State

State is serializable and includes:

- scenario and chat-state objects
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
- `family`

On reset, the environment selects a scenario, initializes the chat state, and asks the simulator to generate the opening customer message.

## Step

`step(action)`:

1. validates and normalizes the action
2. appends the assistant turn
3. applies deterministic hidden-state updates
4. calls the simulator backend for the next customer reply
5. scores the customer reply on the four emotion axes
6. computes reward components
7. checks termination
8. finalizes scorer inputs and trace when done

## Reward

Built-in shaped reward components:

- `score_delta_reward`
- `turn_satisfaction_reward`
- `format_validity_reward`
- `final_satisfaction_reward`

The final scorer remains pluggable.

## Termination Conditions

- `max_turns_reached`
- `objective_achieved`
- `simulator_stopped`
- `invalid_action_threshold_reached`
- `fatal_backend_error`
