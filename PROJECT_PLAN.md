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
- score distribution by action type
- ranking metrics split by target action type
- score gap split by `negative_strength`

## Phase 4: Optional RL Integration

Only start this phase if offline ranking is clearly better than simple
baselines.

Low-risk integration order:

1. rollout filtering;
2. trajectory reranking;
3. small-weight reward shaping.

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
