#!/usr/bin/env bash
# Fast SigLIP2 feature extraction — single GPU, single process.
#
#   bash scripts/run_siglip2_extract_full_train_single_gpu_fast.sh   # 500k train
#   GPU=0 PREFIX=train bash scripts/run_siglip2_extract_fast.sh
#   MAX_SAMPLES=10 ALLOW_LOCAL=1 bash scripts/run_siglip2_extract_fast.sh
#
# Resume-safe: re-run skips finished pairs. Use NOHUP=1 for background (default for train).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_env.sh"

PREFIX="${PREFIX:-eval}"
GPU="${GPU:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
START_INDEX="${START_INDEX:-0}"
AUDIO_BATCH="${AUDIO_BATCH:-10}"
VIDEO_BATCH="${VIDEO_BATCH:-32}"
VIDEO_WORKERS="${VIDEO_WORKERS:-2}"
NO_VIDEO_PREFETCH="${NO_VIDEO_PREFETCH:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
NOHUP="${NOHUP:-0}"
PREFLIGHT="${PREFLIGHT:-1}"
ALLOW_LOCAL="${ALLOW_LOCAL:-0}"

export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p results
LOG="${ROOT}/results/siglip2_${PREFIX}_fast_gpu${GPU}.log"

if [[ "${PREFLIGHT}" == "1" ]]; then
  echo "Preflight CUDA on GPU ${GPU} ..."
  bash "${ROOT}/scripts/probe_gpus.sh" "${GPU}"
fi

SPLIT="$(python -c "from paths_config import PREFIX_CONFIG; print(PREFIX_CONFIG['${PREFIX}']['data_split'])")"
SCP="$(python -c "from paths_config import raw_scp_path, PREFIX_CONFIG; print(raw_scp_path(PREFIX_CONFIG['${PREFIX}']['data_split']))")"
if [[ ! -f "${SCP}" ]]; then
  python preprocess_all.py --splits "${SPLIT}"
fi

SCP_TOTAL="$(wc -l < "${SCP}")"
if [[ "${MAX_SAMPLES}" -gt 0 && "${MAX_SAMPLES}" -lt "${SCP_TOTAL}" ]]; then
  WORK_TOTAL="${MAX_SAMPLES}"
else
  WORK_TOTAL="${SCP_TOTAL}"
fi

: > "${LOG}"

EXTRA=()
[[ "${MAX_SAMPLES}" -gt 0 ]] && EXTRA+=(--max_samples "${MAX_SAMPLES}")
[[ "${START_INDEX}" -gt 0 ]] && EXTRA+=(--start_index "${START_INDEX}")
[[ "${NO_VIDEO_PREFETCH}" == "1" ]] && EXTRA+=(--no_video_prefetch)
[[ "${REQUIRE_CUDA}" == "1" ]] && EXTRA+=(--require_cuda)
[[ "${ALLOW_LOCAL}" == "1" ]] && EXTRA+=(--allow-local)

SEED=$(( 42 + START_INDEX ))

echo "=== Fast SigLIP2 extraction (single GPU) ==="
echo "gpu=${GPU}  prefix=${PREFIX}  scp_lines=${SCP_TOTAL}  work=${WORK_TOTAL}"
echo "start_index=${START_INDEX}  audio_batch=${AUDIO_BATCH}  video_batch=${VIDEO_BATCH}"
echo "log=${LOG}"

cd data_prepare
RUN=(
  python feature_extractor_TVA2_siglip2_fast.py
  --prefix "${PREFIX}"
  --audio_batch_size "${AUDIO_BATCH}"
  --video_batch_size "${VIDEO_BATCH}"
  --video_workers "${VIDEO_WORKERS}"
  --start_index "${START_INDEX}"
  --seed "${SEED}"
  --log_file "${LOG}"
  "${EXTRA[@]}"
)

if [[ "${NOHUP}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" nohup "${RUN[@]}" >/dev/null 2>&1 &
  echo "Started PID $! in background."
  echo "Monitor: tail -f ${LOG}"
  if [[ "${PREFIX}" == "train" && "${WORK_TOTAL}" -ge "${SCP_TOTAL}" ]]; then
    echo "When done: python scripts/rebuild_shuf_scp.py --prefix train && bash run_train.sh"
  fi
else
  CUDA_VISIBLE_DEVICES="${GPU}" "${RUN[@]}"
fi
