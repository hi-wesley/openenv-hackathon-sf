# Scenario Data

The dataset is split into `train`, `val`, and `test` JSONL files. Each line is a complete scenario object with these fields:

- `scenario_id`
- `split`
- `family`
- `title`
- `visible_context`
- `initial_user_message`
- `task_success_criteria`
- `allowed_turns`
- `difficulty`
- `latent_preference_overrides`
- `simulator_instructions`
- `tags`

Split policy:

- `train`: 48 scenarios
- `val`: 12 scenarios
- `test`: 12 scenarios

Each of the six scenario families has 8 train / 2 val / 2 test examples.

