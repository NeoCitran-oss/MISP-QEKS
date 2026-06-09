#!/usr/bin/env bash
# Full train SigLIP2 feature extraction on 6 GPUs (fast path).
#
#   bash scripts/run_siglip2_extract_full_train_fast.sh
#
# Expect ~0.5–1.5 s/pair on GPU (vs ~3 s baseline). All 6 shards run in parallel.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPUS="${GPUS:-0,1,2,3,4,5}" \
MAX_SAMPLES=0 \
PREFIX=train \
AUDIO_BATCH=10 \
VIDEO_BATCH=32 \
VIDEO_WORKERS=2 \
NO_VIDEO_PREFETCH=0 \
LAUNCH_DELAY=20 \
bash scripts/run_siglip2_extract_multi_gpu_fast.sh