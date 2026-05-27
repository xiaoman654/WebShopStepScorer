#!/usr/bin/env bash
set -euo pipefail

python scripts/data/build_scorer_hardneg_v2.py \
  --train-states data/processed/scorer_baseline/train_states.jsonl \
  --valid-states data/processed/scorer_baseline/valid_states.jsonl \
  --out-dir data/processed/scorer_hardneg_v2 \
  --valid-ratio 0.1 \
  --seed 42 \
  --max-history-steps 4 \
  --negatives-per-positive 5
