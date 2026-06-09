#!/usr/bin/env bash
# Launch one fast-extraction shard (for restarts after a failed GPU).
#
#   GPU=1 START=83334 COUNT=83334 bash scripts/run_siglip2_fast_one_shard.sh
#   GPU=0 START=0 COUNT=83334 bash scripts/run_siglip2_fast_one_shard.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_env.sh"

GPU="${GPU:?Set GPU=}"
START="${START:?Set START=}"
COUNT="${COUNT:?Set COUNT=}"
PREFIX="${PREFIX:-train}"
AUDIO_BATCH="${AUDIO_BATCH:-10}"
VIDEO_BATCH="${VIDEO_BATCH:-32}"
VIDEO_WORKERS="${VIDEO_WORKERS:-2}"
NO_VIDEO_PREFETCH="${NO_VIDEO_PREFETCH:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
mkdir -p results

LOG="${ROOT}/results/siglip2_${PREFIX}_fast_gpu${GPU}.log"
SEED=$(( 42 + START ))
: > "${LOG}"

EXTRA=()
[[ "${NO_VIDEO_PREFETCH}" == "1" ]] && EXTRA+=(--no_video_prefetch)
[[ "${REQUIRE_CUDA}" == "1" ]] && EXTRA+=(--require_cuda)

echo "Starting shard GPU=${GPU} start=${START} count=${COUNT} log=${LOG}"
cd data_prepare
CUDA_VISIBLE_DEVICES="${GPU}" nohup python feature_extractor_TVA2_siglip2_fast.py \
  --prefix "${PREFIX}" \
  --audio_batch_size "${AUDIO_BATCH}" \
  --video_batch_size "${VIDEO_BATCH}" \
  --video_workers "${VIDEO_WORKERS}" \
  --start_index "${START}" \
  --max_samples "${COUNT}" \
  --seed "${SEED}" \
  --log_file "${LOG}" \
  --no_rebuild_scp \
  "${EXTRA[@]}" \
  >/dev/null 2>&1 &
echo "PID $! — tail -f ${LOG}"
