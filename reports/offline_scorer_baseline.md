# Offline Scorer Baseline

Status: offline scorer milestone completed

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

## Learned Scorers

After non-model baselines, we evaluated two learned scorers:

1. TF-IDF + logistic regression.
2. Qwen2.5-1.5B LoRA Yes/No scorer.

The Yes/No scorer uses:

```text
score(s, a) = P(Yes) / (P(Yes) + P(No))
```

for each available action, then ranks actions by this score.

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

## Learned Scorer Results

| Scorer | Top-1 | Top-3 | MRR | Mean rank |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.2599 | 0.5294 | 0.4475 | 5.820 |
| Qwen2.5-1.5B Yes/No LoRA | 0.3211 | 0.5851 | 0.4995 | 4.820 |

The TF-IDF scorer has classification signal but does not meaningfully beat the
best simple ranking baseline. The Qwen Yes/No scorer clearly improves ranking:

```text
Top-1: 0.2591 -> 0.3211
Top-3: 0.5469 -> 0.5851
MRR:   0.4455 -> 0.4995
```

This confirms that a semantic cross-encoder scorer learns useful WebShop action
preference beyond action frequency and hand-written action-type priors.

## Qwen Yes/No Scorer By Target Action Type

| Type | states | Top-1 | Top-3 | MRR | Mean rank |
|---|---:|---:|---:|---:|---:|
| buy | 157 | 0.7070 | 0.8599 | 0.7893 | 4.764 |
| click_other | 226 | 0.2965 | 0.5487 | 0.4694 | 7.199 |
| info | 125 | 0.0640 | 0.6640 | 0.3814 | 3.544 |
| item_click | 321 | 0.1682 | 0.4486 | 0.3806 | 4.340 |
| navigation | 71 | 0.0282 | 0.2254 | 0.2193 | 7.042 |
| option | 1 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| pagination | 357 | 0.4510 | 0.6527 | 0.5939 | 3.784 |

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

The first offline scorer milestone is successful. The Qwen Yes/No scorer beats
the strongest non-model baseline and TF-IDF baseline on overall ranking.

Next steps should not jump directly to reward shaping. Recommended order:

```text
1. Extract qualitative success/failure cases from Qwen scorer ranking.
2. Compare scorer behavior by action type, especially item_click and navigation.
3. Build a reranking or filtering prototype before using the scorer as reward.
```
