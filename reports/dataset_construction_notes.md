# Dataset Construction Notes

## Scope

The first dataset is a discrete admissible-action preference dataset.

For each WebShop state, the builder keeps the state only if:

```text
target_action in available_actions
available_actions has at least one contrastive alternative
```

This means free-form search generation is intentionally excluded when the
demonstrated `search[...]` action does not appear in the admissible action list.

## Labels

The labels are preference-style labels:

```text
label = 1: demonstrated preferred action
label = 0: sampled contrastive alternative
```

The negative label does not mean an action is absolutely wrong. WebShop can have
multiple locally reasonable actions, especially among information-gathering
actions such as:

```text
click[description]
click[features]
click[reviews]
```

## Negative Sampling

For each kept state, the dataset builder creates:

```text
1 positive example
up to 3 negative examples
```

Negative examples are sampled from the same state's `available_actions` list.
Each negative has:

```text
negative_strength = hard | weak | random
```

Current heuristic:

- `hard`: same action type as the target, except ambiguous navigation/info
  types.
- `weak`: ambiguous alternatives such as info/navigation/pagination actions.
- `random`: different action type alternatives.

## Action Types

The builder records both target and candidate action types:

```text
search
buy
pagination
navigation
info
item_click
option
click_other
other
```

Offline evaluation should report metrics both overall and by target action type.

## Output Files

Classification examples:

```text
train.jsonl
valid.jsonl
```

State-level ranking rows:

```text
train_states.jsonl
valid_states.jsonl
```

Construction statistics:

```text
stats.json
```

The `*_states.jsonl` files should be used for ranking evaluation. For each
state, score every available action and compute the rank of the demonstrated
target action.
