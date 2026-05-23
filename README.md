# WebShopStepScorer

Step-level action scoring for WebShop agents.

This project is a follow-up to the WarmGiGPO-WebShop SFT + GiGPO project. The
first milestone is deliberately offline: build a scorer that ranks candidate
WebShop actions at a given state before integrating it into RL.

## Phase 1 Goal

Use WebShop human demonstrations to train and evaluate an action scorer:

```text
task + history + current observation + candidate action -> quality score
```

The scorer is considered useful only if it can rank the human-demonstration
target action near the top among the current state's admissible actions.

## Initial Scope

In scope:

- Parse existing WebShop human demonstrations.
- Construct positive and negative step-level action-scoring examples.
- Train a lightweight scorer with BCE / classification loss.
- Evaluate offline ranking metrics: Top-1, Top-k, MRR, AUC, score gap.

Out of scope for the first milestone:

- Online GiGPO integration.
- Reward shaping.
- Strong-model rollout generation.
- Multi-teacher ablations.

## Repository Layout

```text
configs/        Experiment configs
scripts/data/   Dataset construction and inspection
scripts/train/  Scorer training
scripts/eval/   Offline scorer evaluation
reports/        Notes, tables, and final writeups
```

## First Success Criterion

The first checkpoint is an offline report showing whether the scorer can rank
the demonstrated action above sampled negative actions from the same
`available_actions` set.
