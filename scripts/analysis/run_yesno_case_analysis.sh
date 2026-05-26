#!/usr/bin/env bash
set -euo pipefail

python scripts/analysis/extract_yesno_ranking_cases.py \
  --ranking-json data/processed/scorer_baseline/yesno_scorer_full_ranking.json \
  --states data/processed/scorer_baseline/valid_states.jsonl \
  --out-md reports/yesno_scorer_case_analysis.md \
  --limit 5 \
  --obs-chars 1200 \
  --history-chars 400
