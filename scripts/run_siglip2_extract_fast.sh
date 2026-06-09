#!/usr/bin/env bash
# Fast SigLIP2 feature extraction (single process).
#
#   source scripts/activate_env.sh
#   MAX_SAMPLES=10 bash scripts/run_siglip2_extract_fast.sh
#   PREFIX=train MAX_SAMPLES=50000 bash scripts/run_siglip2_extract_fast.sh
#
# Multi-GPU 50k (recommended on tars):
#   GPUS="0,4,5,6" bash scripts/run_siglip2_extract_multi_gpu_fast.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_env.sh"

PREFIX="${PREFIX:-eval}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
AUDIO_BATCH="${AUDIO_BATCH:-8}"
VIDEO_BATCH="${VIDEO_BATCH:-32}"
VIDEO_WORKERS="${VIDEO_WORKERS:-4}"
NOHUP="${NOHUP:-0}"

mkdir -p results
LOG="results/siglip2_${PREFIX}_fast.log"

SPLIT="$(python -c "from paths_config import PREFIX_CONFIG; print(PREFIX_CONFIG['${PREFIX}']['data_split'])")"
SCP="$(python -c "from paths_config import raw_scp_path, PREFIX_CONFIG; print(raw_scp_path(PREFIX_CONFIG['${PREFIX}']['data_split']))")"
if [[ ! -f "${SCP}" ]]; then
  python preprocess_all.py --splits "${SPLIT}"
fi

EXTRA=()
[[ "${MAX_SAMPLES}" -gt 0 ]] && EXTRA+=(--max_samples "${MAX_SAMPLES}")

echo "=== Fast SigLIP2 extraction ==="
echo "prefix=${PREFIX} audio_batch=${AUDIO_BATCH} video_batch=${VIDEO_BATCH} workers=${VIDEO_WORKERS}"

cd data_prepare
RUN=(python feature_extractor_TVA2_siglip2_fast.py
  --prefix "${PREFIX}"
  --audio_batch_size "${AUDIO_BATCH}"
  --video_batch_size "${VIDEO_BATCH}"
  --video_workers "${VIDEO_WORKERS}"
  --log_file "../${LOG}"
  "${EXTRA[@]}"
)

if [[ "${NOHUP}" == "1" ]]; then
  nohup "${RUN[@]}" >/dev/null 2>&1 &
  echo "Started PID $! | tail -f ${ROOT}/${LOG}"
else
  "${RUN[@]}"
fi

echo "Log: ${ROOT}/${LOG}"
