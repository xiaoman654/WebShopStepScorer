#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent-webshop
source /etc/network_turbo || true

export OMP_NUM_THREADS=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_DIR=${MODEL_DIR:-/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
PROJECT_DIR=${PROJECT_DIR:-/root/autodl-fs/WebShopStepScorer}

cd "$PROJECT_DIR"

python scripts/eval/evaluate_yesno_scorer_ranking.py \
  --base-model "$MODEL_DIR" \
  --adapter outputs/yesno_scorer/qwen25_1p5b_lora_smoke/final_adapter \
  --states data/processed/scorer_baseline/valid_states.jsonl \
  --out-json data/processed/scorer_baseline/yesno_scorer_smoke_ranking.json \
  --out-md reports/yesno_scorer_smoke_ranking.md \
  --max-states 100 \
  --batch-size 8 \
  --max-seq-length 2048 \
  --attn-implementation flash_attention_2 \
  --bf16
