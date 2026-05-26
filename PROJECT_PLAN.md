# WebShopStepScorer Project Plan

## Project Definition

Train a step-level WebShop action scorer from human demonstration trajectories,
then use offline ranking metrics to decide whether it is worth integrating into
GiGPO.

## Research Question

Can a model learn local WebShop action quality from demonstration states and
admissible actions?

More concretely:

```text
Given task, history, current observation, and all admissible actions,
can a scorer rank the human-demonstration target action above alternatives?
```

## Why Start Offline

Directly adding a learned reward to GiGPO is risky because:

- step-level labels are noisy;
- reward shaping can cause reward hacking;
- teacher/student rollout distributions may differ;
- RL experiments are expensive and slower to debug.

The offline scorer is a lower-risk first milestone. If it fails ranking
evaluation, it should not be used for RL.

## Phase 1: Human-Demo Scorer Dataset

Input source:

```text
WebShop human demonstration trajectories
```

Each step should produce:

```text
instruction
history
observation
available_actions
target_action
```

Training examples:

```text
positive: candidate_action = target_action, label = 1
negative: candidate_action in available_actions except target_action, label = 0
```

Important terminology:

```text
label=1 means demonstrated preferred action.
label=0 means sampled contrastive alternative.
```

The negative label should not be interpreted as "absolutely wrong". WebShop
often has multiple locally reasonable actions, such as `description`,
`features`, and `reviews`. The first version trains an imitation/preference
scorer, not a perfect reward model.

Recommended negative sampling:

- include 1-3 random negatives per state;
- include hard negatives when possible, such as actions with the same type:
-  `click[item - ...]` vs `click[item - ...]`;
- keep a `negative_strength` field: `hard`, `weak`, or `random`;
- exclude exact duplicate actions.

Search boundary:

The first version only covers discrete admissible-action ranking. If a
free-form `search[...]` target is not explicitly present in `available_actions`,
the step is skipped. Search query generation can be handled as a separate
project later.

## Phase 2: Scorer Model

Initial model objective:

```text
BCE classification over (state, candidate_action)
```

Input template:

```text
Task:
{instruction}

Recent history:
{history}

Current observation:
{observation}

Candidate action:
{candidate_action}

Question:
Is this candidate action a good next action for the task?
```

Output:

```text
score in [0, 1]
```

## Phase 3: Offline Evaluation

For each held-out state:

1. score every admissible action;
2. sort by score;
3. compute the rank of the human target action.

Metrics:

- Top-1 accuracy
- Top-3 accuracy
- MRR
- AUC
- positive/negative score gap
- target vs. top-1 score margin
- whether the top-1 action has the same action type as the target
- score distribution by action type
- ranking metrics split by target action type
- score gap split by `negative_strength`

Qualitative analysis should be standardized rather than purely anecdotal. Each
case report should include:

- target action rank, score, type, and original action-list position;
- top-1/top-3 actions, scores, types, and score margins;
- number of admissible actions;
- observation/history lengths;
- whether the top-1 action shares the target action type;
- an error taxonomy label.

Initial error taxonomy:

- `type_confusion`: top-1 and target have different action types;
- `within_type_item_confusion`: both actions are item clicks, but the item differs;
- `attribute_or_entity_mismatch`: likely fine-grained product/attribute mismatch;
- `position_bias`: scorer prefers an earlier action in the action list;
- `generic_info_bias`: scorer over-prefers `description`, `features`, or `reviews`;
- `late_stage_buy_bias`: scorer over-prefers `buy now`.

## Phase 4: Offline Selector Utility

Before RL integration, run an offline selector simulation:

1. use the scorer top-1 action as the selected action for each validation state;
2. compare exact match, top-k inclusion, same-type match, and score margin;
3. split results by action type and error taxonomy.

This phase answers whether the scorer is useful as an action selector or
filter, not merely whether it has a reasonable AUC.

## Phase 5: Harder Negatives and Objective Ablations

If the first scorer mostly fails on fine-grained item selection, add a smaller
hard-negative dataset rather than only scaling data size. Useful harder
negatives include:

- same-page item clicks with similar titles;
- same category but wrong color, size, price, or brand;
- `description` / `features` / `reviews` weak negatives;
- buy actions from states where a required attribute is still missing.

The first training objective is pointwise Yes/No scoring. A later small
comparison can test pairwise ranking:

```text
state + action_a + action_b -> which action is better?
```

Pairwise ranking is more aligned with the final selector use case, but it should
remain a follow-up until the pointwise scorer's failure modes are understood.

## Phase 6: Optional RL Integration

Only start this phase if offline ranking is clearly better than simple
baselines and selector-simulation errors are explainable.

Low-risk integration order:

1. action reranking / selector prototype;
2. rollout filtering;
3. trajectory reranking;
4. small-weight reward shaping.

Reward shaping should be last:

```text
r'_t = r_t + lambda * scorer(s_t, a_t)
```

with a small `lambda`.

## First Deliverable

Create a report:

```text
reports/offline_scorer_baseline.md
```

It should answer:

- how many states/actions were used;
- how positives and negatives were constructed;
- whether the scorer beats random ranking;
- what failure cases look like;
- whether the project should proceed to RL integration.
