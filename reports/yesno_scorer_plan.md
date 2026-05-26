# Yes/No Qwen Scorer Plan

## Motivation

The TF-IDF scorer achieved AUC 0.6906 but did not meaningfully improve in-state
ranking over simple action priors:

```text
TF-IDF Top-1 = 0.2599
TF-IDF Top-3 = 0.5294
TF-IDF MRR   = 0.4475
```

It mainly learned easy action types such as `buy` and `pagination`, while
remaining weak on `item_click`, `info`, `navigation`, and `click_other`.

This motivates a semantic cross-encoder scorer.

## Formulation

Convert each scorer row into a chat-style yes/no decision:

```text
User:
Task
Recent history
Current observation
Candidate action
Candidate action type
Is this candidate action a good next action?

Assistant:
Yes
```

Positive rows use `Yes`; sampled contrastive alternatives use `No`.

At evaluation time, for each state and each available action, compute:

```text
P(Yes) / (P(Yes) + P(No))
```

Then rank actions by this score.

## First Smoke

Use:

```text
max_train_samples = 2000
max_valid_samples = 500
max_steps = 80
```

Evaluate first on 100 validation states. If the result is sane, evaluate all
1258 validation states or run a full 1-epoch scorer.

## Success Threshold

The scorer should beat the strongest simple baselines:

```text
Top-1 > 0.259
Top-3 > 0.547
MRR   > 0.446
```

It should also improve non-trivial action types, especially:

```text
item_click
click_other
info
navigation
```

## Full Result

The full Qwen2.5-1.5B Yes/No LoRA scorer was trained on the complete scorer SFT
set and evaluated on all 1258 validation states.

| Scorer | Top-1 | Top-3 | MRR | Mean rank |
|---|---:|---:|---:|---:|
| action_text_prior | 0.2591 | 0.5469 | 0.4455 | 5.517 |
| TF-IDF + Logistic Regression | 0.2599 | 0.5294 | 0.4475 | 5.820 |
| Qwen2.5-1.5B Yes/No LoRA | 0.3211 | 0.5851 | 0.4995 | 4.820 |

The result passes the success threshold. The scorer is not a return-based
critic yet, but it is a useful critic-like action preference model:

```text
f(s, a) ~= probability that a candidate action matches the demonstrated
preferred action under the current WebShop state.
```

Recommended next step: qualitative ranking-case analysis and reranking/filtering
prototype. Do not use it as reward shaping before checking reward-hacking risk.
