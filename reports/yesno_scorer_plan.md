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
