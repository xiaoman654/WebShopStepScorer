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

## Build the First Scorer Dataset

On the server, place or reference the WebShop human demonstration file:

```text
/root/autodl-fs/WarmGiGPO-WebShop/data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl
```

Then run:

```bash
cd /root/autodl-fs/WebShopStepScorer

python scripts/data/build_scorer_dataset.py \
  --input /root/autodl-fs/WarmGiGPO-WebShop/data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl \
  --out-dir data/processed/scorer_baseline \
  --valid-ratio 0.1 \
  --seed 42 \
  --max-history-steps 4 \
  --negatives-per-positive 3
```

This writes:

```text
data/processed/scorer_baseline/train.jsonl
data/processed/scorer_baseline/valid.jsonl
data/processed/scorer_baseline/train_states.jsonl
data/processed/scorer_baseline/valid_states.jsonl
data/processed/scorer_baseline/stats.json
```

The first dataset intentionally skips states where the demonstrated target
action is not in `available_actions`. This makes the first milestone a discrete
admissible-action ranking task, not a free-form search-query generation task.
