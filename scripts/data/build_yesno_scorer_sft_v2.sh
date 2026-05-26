#!/usr/bin/env bash
set -euo pipefail

python scripts/data/build_yesno_scorer_sft.py \
  --train data/processed/scorer_hardneg_v2/train.jsonl \
  --valid data/processed/scorer_hardneg_v2/valid.jsonl \
  --out-dir data/processed/yesno_scorer_sft_v2_hardneg \
  --max-observation-chars 4000 \
  --max-history-chars 800
