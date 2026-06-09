#!/usr/bin/env bash
# Full train (500k) SigLIP2 fast extraction on ONE GPU.
#
#   bash scripts/run_siglip2_extract_full_train_single_gpu_fast.sh
#   GPU=0 bash scripts/run_siglip2_extract_full_train_single_gpu_fast.sh
#
# Resume-safe. Typical wall time: ~4–7 days on one GPU at ~1 s/pair (skips are faster).
# Run inside tmux/screen so SSH disconnect is OK (uses NOHUP=1).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}" \
PREFIX=train \
MAX_SAMPLES=0 \
START_INDEX=0 \
NOHUP=1 \
PREFLIGHT=1 \
bash scripts/run_siglip2_extract_fast.sh
