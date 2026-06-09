#!/usr/bin/env bash
# Fast SigLIP2 feature extraction (batched audio + decord video prefetch).
#
#   source scripts/activate_env.sh
#   MAX_SAMPLES=10 bash scripts/run_siglip2_extract_fast.sh
#
# Multi-GPU: launch one process per GPU with disjoint --start_index / --max_samples
# (same pattern as run_siglip2_extract_multi_gpu.sh but call feature_extractor_TVA2_siglip2_fast.py)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/config/paths.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/config/paths.env"
fi
source "${ROOT}/envs/misp-qeks/bin/activate"

PREFIX="${PREFIX:-eval}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
AUDIO_BATCH="${AUDIO_BATCH:-8}"
VIDEO_BATCH="${VIDEO_BATCH:-32}"
VIDEO_WORKERS="${VIDEO_WORKERS:-4}"
ALLOW_LOCAL="${ALLOW_LOCAL:-1}"

mkdir -p results
LOG="results/siglip2_${PREFIX}_fast.log"

EXTRA=()
[[ "${MAX_SAMPLES}" -gt 0 ]] && EXTRA+=(--max_samples "${MAX_SAMPLES}")
[[ "${ALLOW_LOCAL}" == "1" ]] && EXTRA+=(--allow-local)

echo "=== Fast SigLIP2 extraction ==="
echo "prefix=${PREFIX} audio_batch=${AUDIO_BATCH} video_batch=${VIDEO_BATCH} workers=${VIDEO_WORKERS}"

python preprocess_all.py --splits "$(python -c "from paths_config import PREFIX_CONFIG; print(PREFIX_CONFIG['${PREFIX}']['data_split'])")"

cd data_prepare
python feature_extractor_TVA2_siglip2_fast.py \
  --prefix "${PREFIX}" \
  --audio_batch_size "${AUDIO_BATCH}" \
  --video_batch_size "${VIDEO_BATCH}" \
  --video_workers "${VIDEO_WORKERS}" \
  --log_file "../${LOG}" \
  "${EXTRA[@]}"

echo "Log: ${ROOT}/${LOG}"
