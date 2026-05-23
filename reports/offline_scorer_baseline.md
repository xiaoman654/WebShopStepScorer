# Offline Scorer Baseline

Status: not started

## Goal

Evaluate whether a step-level action scorer can rank the human-demonstration
target action above other admissible actions in WebShop states.

## Dataset

TODO:

- source trajectories:
- train states:
- valid states:
- positives:
- negatives:
- average admissible actions per state:

## Model

TODO:

- model:
- objective:
- max sequence length:
- training steps:

## Metrics

TODO:

| Metric | Value |
|---|---:|
| Top-1 | TBD |
| Top-3 | TBD |
| MRR | TBD |
| AUC | TBD |
| Positive-negative score gap | TBD |

## Decision

TODO:

Proceed to RL integration only if the scorer clearly beats random ranking and
shows meaningful qualitative behavior on held-out states.
