# Scenario Data

The environment now centers on three customer-service scenario families:

- `late_delivery_refund`
- `damaged_item_replacement`
- `surprise_billing_cancellation`

Each JSONL row contains:

- `scenario_id`
- `split`
- `family`
- `title`
- `visible_context`
- `initial_user_message`
  This is a short seed that the simulator uses to generate the opening customer message.
- `task_success_criteria`
- `allowed_turns`
- `difficulty`
- `simulator_instructions`
- `tags`

Each split contains one scenario per family for a total of three per split.
