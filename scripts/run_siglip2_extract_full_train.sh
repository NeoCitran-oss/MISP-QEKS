#!/usr/bin/env bash
# Full train SigLIP2 feature extraction on 6 GPUs (500k pairs, baseline extractor).
#
#   bash scripts/run_siglip2_extract_full_train.sh
#
# Resume-safe: re-run skips finished pairs. Use tmux/SSH disconnect OK (nohup).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPUS="${GPUS:-0,1,2,3,4,5}" \
MAX_SAMPLES=0 \
PREFIX=train \
BATCH_SIZE="${BATCH_SIZE:-16}" \
bash scripts/run_siglip2_extract_multi_gpu.sh
