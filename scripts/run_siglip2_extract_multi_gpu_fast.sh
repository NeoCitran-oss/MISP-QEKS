#!/usr/bin/env bash
# Fast parallel SigLIP2 extraction (batched Qwen + decord/prefetch video).
#
#   bash scripts/run_siglip2_extract_full_train_fast.sh   # 500k train, GPUs 0-5
#   GPUS="0,1,2,3,4,5" MAX_SAMPLES=0 bash scripts/run_siglip2_extract_multi_gpu_fast.sh
#   GPUS="0,1,4,5" MAX_SAMPLES=50000 bash scripts/run_siglip2_extract_multi_gpu_fast.sh
#
# After ALL shards finish (full train):
#   python scripts/rebuild_shuf_scp.py --prefix train
#   bash run_train.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_env.sh"

PREFIX="${PREFIX:-train}"
GPUS="${GPUS:-0,1,2,3,4,5}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
AUDIO_BATCH="${AUDIO_BATCH:-8}"
VIDEO_BATCH="${VIDEO_BATCH:-32}"
VIDEO_WORKERS="${VIDEO_WORKERS:-2}"
NO_VIDEO_PREFETCH="${NO_VIDEO_PREFETCH:-0}"
LAUNCH_DELAY="${LAUNCH_DELAY:-0}"
WAIT_FOR_CUDA_READY="${WAIT_FOR_CUDA_READY:-1}"
MODEL_LOAD_TIMEOUT="${MODEL_LOAD_TIMEOUT:-600}"
KILL_OLD="${KILL_OLD:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
RESET_LOGS="${RESET_LOGS:-1}"
PREFLIGHT_GPUS="${PREFLIGHT_GPUS:-1}"
mkdir -p results

wait_for_shard_ready() {
  local log_path="$1"
  local timeout_sec="$2"
  local elapsed=0
  while [[ "${elapsed}" -lt "${timeout_sec}" ]]; do
    if [[ -f "${log_path}" ]] && grep -q "Using: cuda" "${log_path}" \
        && grep -q "SigLIP2 output dim:" "${log_path}"; then
      echo "  GPU ready (${elapsed}s): ${log_path}"
      return 0
    fi
    if [[ -f "${log_path}" ]] && grep -q "ERROR: CUDA unavailable" "${log_path}"; then
      echo "  ERROR: CUDA failed in ${log_path} — stop and fix before launching more GPUs."
      return 1
    fi
    if [[ -f "${log_path}" ]] && grep -q "Using: cpu" "${log_path}"; then
      echo "  ERROR: shard fell back to CPU in ${log_path} (require_cuda should prevent this)."
      return 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
    if (( elapsed % 60 == 0 )); then
      echo "  still loading models (${elapsed}s / ${timeout_sec}s) ..."
    fi
  done
  echo "  WARN: timeout (${timeout_sec}s) waiting for ${log_path}; continuing anyway."
  return 0
}

if [[ "${KILL_OLD}" == "1" ]]; then
  if pgrep -f "feature_extractor_TVA2_siglip2" >/dev/null 2>&1; then
    echo "Stopping existing SigLIP2 extraction jobs ..."
    pkill -f "feature_extractor_TVA2_siglip2" || true
    echo "Waiting for GPU processes to exit ..."
    sleep 10
  fi
fi

export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"

if [[ "${PREFLIGHT_GPUS}" == "1" ]]; then
  echo "Preflight: probing GPUs ${GPUS} (no extraction jobs should be running) ..."
  bash "${ROOT}/scripts/probe_gpus.sh" "${GPU_ARR[@]}"
fi

SPLIT="$(python -c "from paths_config import PREFIX_CONFIG; print(PREFIX_CONFIG['${PREFIX}']['data_split'])")"
SCP="$(python -c "from paths_config import raw_scp_path, PREFIX_CONFIG; print(raw_scp_path(PREFIX_CONFIG['${PREFIX}']['data_split']))")"

if [[ ! -f "${SCP}" ]]; then
  echo "Building raw dicts for split=${SPLIT} ..."
  python preprocess_all.py --splits "${SPLIT}"
fi

SCP_TOTAL="$(wc -l < "${SCP}")"
if [[ "${MAX_SAMPLES}" -gt 0 && "${MAX_SAMPLES}" -lt "${SCP_TOTAL}" ]]; then
  WORK_TOTAL="${MAX_SAMPLES}"
else
  WORK_TOTAL="${SCP_TOTAL}"
fi

NUM_GPUS="${#GPU_ARR[@]}"
CHUNK=$(( (WORK_TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

echo "=== Fast SigLIP2 multi-GPU extraction ==="
echo "prefix=${PREFIX}  scp_lines=${SCP_TOTAL}  extract_total=${WORK_TOTAL}"
echo "gpus=${GPUS}  chunk≈${CHUNK}  wait_for_cuda=${WAIT_FOR_CUDA_READY}  reset_logs=${RESET_LOGS}"
echo "audio_batch=${AUDIO_BATCH}  video_batch=${VIDEO_BATCH}  video_workers=${VIDEO_WORKERS}  no_prefetch=${NO_VIDEO_PREFETCH}"
echo "Tip: next GPU starts only after prior log shows 'Using: cuda' + 'SigLIP2 output dim'."
cd data_prepare
PIDS=()
PREV_LOG=""

for idx in "${!GPU_ARR[@]}"; do
  GPU="${GPU_ARR[$idx]}"
  if [[ "${idx}" -gt 0 ]]; then
    if [[ "${WAIT_FOR_CUDA_READY}" == "1" && -n "${PREV_LOG}" ]]; then
      PREV_GPU="${GPU_ARR[$((idx - 1))]}"
      echo "Waiting for GPU ${PREV_GPU} to finish model load before starting GPU ${GPU} ..."
      if ! wait_for_shard_ready "${PREV_LOG}" "${MODEL_LOAD_TIMEOUT}"; then
        echo "Aborting multi-GPU launch. Fix CUDA on failed GPU, then restart only that shard:"
        echo "  bash scripts/probe_gpus.sh ${PREV_GPU}"
        echo "  GPU=${PREV_GPU} START=... COUNT=... bash scripts/run_siglip2_fast_one_shard.sh"
        exit 1
      fi
      if [[ "${LAUNCH_DELAY}" -gt 0 ]]; then
        sleep "${LAUNCH_DELAY}"
      fi
    elif [[ "${LAUNCH_DELAY}" -gt 0 ]]; then
      echo "Stagger launch: sleeping ${LAUNCH_DELAY}s before GPU ${GPU} ..."
      sleep "${LAUNCH_DELAY}"
    fi
  fi
  START=$(( idx * CHUNK ))
  if [[ "${START}" -ge "${WORK_TOTAL}" ]]; then
    echo "Skip GPU ${GPU}: start_index ${START} >= ${WORK_TOTAL}"
    continue
  fi
  if [[ "${idx}" -eq $((NUM_GPUS - 1)) ]]; then
    COUNT=$(( WORK_TOTAL - START ))
  else
    COUNT="${CHUNK}"
  fi

  LOG="${ROOT}/results/siglip2_${PREFIX}_fast_gpu${GPU}.log"
  SEED=$(( 42 + START ))

  if [[ "${RESET_LOGS}" == "1" ]]; then
    : > "${LOG}"
  fi

  EXTRA=()
  [[ "${NO_VIDEO_PREFETCH}" == "1" ]] && EXTRA+=(--no_video_prefetch)
  [[ "${REQUIRE_CUDA}" == "1" ]] && EXTRA+=(--require_cuda)

  echo "GPU ${GPU}: start_index=${START} max_samples=${COUNT} log=${LOG}"
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
  PIDS+=("$!")
  echo "  PID ${PIDS[-1]}"
  PREV_LOG="${LOG}"
done

echo ""
echo "Started ${#PIDS[@]} fast workers."
echo "Monitor:"
for idx in "${!GPU_ARR[@]}"; do
  GPU="${GPU_ARR[$idx]}"
  echo "  tail -f ${ROOT}/results/siglip2_${PREFIX}_fast_gpu${GPU}.log"
done
echo ""
if [[ "${WORK_TOTAL}" -ge "${SCP_TOTAL}" ]]; then
  echo "When all are done:"
  echo "  python scripts/rebuild_shuf_scp.py --prefix train"
  echo "  bash run_train.sh"
else
  echo "When all are done:"
  echo "  python scripts/build_partial_train_scp.py --max_pairs ${WORK_TOTAL}"
  echo "  bash run_train_siglip2_quick.sh"
fi