#!/usr/bin/env bash
set -euo pipefail

python scripts/data/build_yesno_scorer_sft.py \
  --train data/processed/scorer_baseline/train.jsonl \
  --valid data/processed/scorer_baseline/valid.jsonl \
  --out-dir data/processed/yesno_scorer_sft \
  --max-observation-chars 4000 \
  --max-history-chars 800
