#!/usr/bin/env bash
set -euo pipefail

python scripts/data/build_scorer_hardneg_v2.py \
  --input data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl \
  --out-dir data/processed/scorer_hardneg_v2 \
  --valid-ratio 0.1 \
  --seed 42 \
  --max-history-steps 4 \
  --negatives-per-positive 5
