#!/usr/bin/env bash
set -euo pipefail

python scripts/eval/evaluate_selector_simulation.py \
  --ranking-json data/processed/scorer_hardneg_v2/yesno_scorer_v2_hardneg_ranking.json \
  --out-json reports/selector_simulation_v2_hardneg.json \
  --out-md reports/selector_simulation_v2_hardneg.md \
  --topk 1,2,3,5
