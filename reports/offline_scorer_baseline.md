# Offline Scorer Baseline

Status: non-model baselines completed; learned scorer pending

## Goal

Evaluate whether a step-level action scorer can rank the human-demonstration
target action above other admissible actions in WebShop states.

## Dataset

- source trajectories: 1571 WebShop human demonstration trajectories
- kept states: 12914
- train states: 11656
- valid states: 1258
- train examples: 44436
- valid examples: 4777
- average admissible actions per valid state: 14.927
- free-form search steps skipped: 2197

## Model

First learned scorer is pending. Before model training, we evaluated non-model
ranking baselines on `valid_states.jsonl`.

## Non-Model Ranking Baselines

| Baseline | Top-1 | Top-3 | MRR | Mean rank |
|---|---:|---:|---:|---:|
| random_expected | 0.1281 | 0.3346 | 0.3069 | 7.963 |
| random_sampled | 0.1288 | 0.3466 | 0.3102 | 8.100 |
| action_type_prior | 0.2591 | 0.4340 | 0.4112 | 7.458 |
| action_text_prior | 0.2591 | 0.5469 | 0.4455 | 5.517 |
| heuristic_action_order | 0.2464 | 0.4046 | 0.3914 | 6.525 |

The strongest simple baseline is `action_text_prior` by MRR and Top-3, while
`action_type_prior` has the same Top-1. A learned scorer should clearly exceed
these numbers before we consider RL integration.

## Action-Type Notes

- `pagination` is easy for action-type prior: Top-1 is 0.9132 because
  pagination states often expose a strong action-type pattern.
- `buy` is trivial for the hand-written heuristic because it ranks `buy now`
  first whenever available, so learned models must be evaluated by action type,
  not only overall.
- `item_click`, `click_other`, `navigation`, and `info` remain much harder and
  are more informative for whether the scorer learns task grounding.
- `option` has only 1 valid state in this split, so no strong conclusion should
  be drawn for option selection yet.

## Decision

Proceed to the first learned scorer. The next baseline should be a lightweight
CPU text model, such as TF-IDF + logistic regression, before moving to an LLM
cross-encoder scorer.

The first learned scorer target is:

```text
Top-1 > 0.259
Top-3 > 0.547
MRR   > 0.446
```
