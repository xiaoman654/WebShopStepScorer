#!/usr/bin/env bash
set -euo pipefail

python scripts/eval/evaluate_selector_simulation.py \
  --ranking-json data/processed/scorer_baseline/yesno_scorer_full_ranking.json \
  --out-json reports/selector_simulation.json \
  --out-md reports/selector_simulation.md \
  --topk 1,2,3,5
