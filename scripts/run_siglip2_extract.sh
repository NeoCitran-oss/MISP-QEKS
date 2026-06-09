#!/usr/bin/env bash
# SigLIP2 TVA feature extraction — single entry point (fast or baseline, 1 or N GPUs).
#
# Common (tars, single GPU — recommended while multi-GPU CUDA is flaky):
#   PREFIX=train NOHUP=1 bash scripts/run_siglip2_extract.sh
#
# Smoke test:
#   MAX_SAMPLES=10 ALLOW_LOCAL=1 bash scripts/run_siglip2_extract.sh
#
# Eval (baseline, slower):
#   FAST=0 PREFIX=eval bash scripts/run_siglip2_extract.sh
#
# Multi-GPU fast (optional):
#   GPUS=0,1,2,3,4,5 PREFIX=train bash scripts/run_siglip2_extract.sh
#
# Resume one shard manually (after multi-GPU partial run):
#   GPU=1 START_INDEX=83334 MAX_SAMPLES=83334 NOHUP=1 bash scripts/run_siglip2_extract.sh
#
# Env vars:
#   FAST=1|0          fast batched path (default 1) vs baseline sequential
#   PREFIX            train|eval|dev|eval_unseen (default train)
#   GPU               single-GPU id (default 0)
#   GPUS              comma list → multi-GPU mode, e.g. 0,1,2,3,4,5
#   MAX_SAMPLES       0 = all (default 0)
#   START_INDEX       scp offset for resume/sharding (default 0)
#   NOHUP=1           background (recommended for long train jobs)
#   PREFLIGHT=1       run probe_gpus.sh before start (default 1)
#   REQUIRE_CUDA=1    exit if CUDA unavailable (default 1)
#   KILL_OLD=1        pkill existing extractors first (default 1 for multi, 0 for single)
#
# After full train: python scripts/rebuild_shuf_scp.py --prefix train && bash run_train.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_env.sh"

# --- defaults ---
FAST="${FAST:-1}"
PREFIX="${PREFIX:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
START_INDEX="${START_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
AUDIO_BATCH="${AUDIO_BATCH:-10}"
VIDEO_BATCH="${VIDEO_BATCH:-32}"
VIDEO_WORKERS="${VIDEO_WORKERS:-2}"
NO_VIDEO_PREFETCH="${NO_VIDEO_PREFETCH:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
NOHUP="${NOHUP:-0}"
PREFLIGHT="${PREFLIGHT:-1}"
ALLOW_LOCAL="${ALLOW_LOCAL:-0}"
RESET_LOGS="${RESET_LOGS:-1}"
WAIT_FOR_CUDA_READY="${WAIT_FOR_CUDA_READY:-1}"
MODEL_LOAD_TIMEOUT="${MODEL_LOAD_TIMEOUT:-600}"
LAUNCH_DELAY="${LAUNCH_DELAY:-0}"

export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- single vs multi ---
if [[ -n "${GPUS:-}" && "${GPUS}" == *","* ]]; then
  MULTI=1
elif [[ "${MULTI:-0}" == "1" ]]; then
  GPUS="${GPUS:-0,1,2,3,4,5}"
  MULTI=1
else
  MULTI=0
  GPU="${GPU:-${GPUS:-0}}"
fi

KILL_OLD="${KILL_OLD:-$([[ "${MULTI}" == "1" ]] && echo 1 || echo 0)}"

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
    if [[ -f "${log_path}" ]] && grep -qE "ERROR: CUDA unavailable|Using: cpu" "${log_path}"; then
      echo "  ERROR: CUDA/CPU failure in ${log_path}"
      return 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
    if (( elapsed % 60 == 0 )); then
      echo "  still loading models (${elapsed}s / ${timeout_sec}s) ..."
    fi
  done
  echo "  WARN: timeout (${timeout_sec}s) waiting for ${log_path}"
  return 0
}

print_when_done() {
  local work_total="$1"
  local scp_total="$2"
  if [[ "${work_total}" -ge "${scp_total}" && "${PREFIX}" == "train" ]]; then
    echo "When done: python scripts/rebuild_shuf_scp.py --prefix train && bash run_train.sh"
  elif [[ "${work_total}" -lt "${scp_total}" ]]; then
    echo "When done: python scripts/build_partial_train_scp.py --max_pairs ${work_total}"
    echo "           bash run_train_siglip2_quick.sh"
  fi
}

if [[ "${KILL_OLD}" == "1" ]] && pgrep -f "feature_extractor_TVA2_siglip2" >/dev/null 2>&1; then
  echo "Stopping existing SigLIP2 jobs ..."
  pkill -f "feature_extractor_TVA2_siglip2" || true
  sleep 10
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

# ======================== MULTI-GPU ========================
if [[ "${MULTI}" == "1" ]]; then
  IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
  NUM_GPUS="${#GPU_ARR[@]}"
  CHUNK=$(( (WORK_TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

  if [[ "${PREFLIGHT}" == "1" ]]; then
    echo "Preflight: probing GPUs ${GPUS} ..."
    bash "${ROOT}/scripts/probe_gpus.sh" "${GPU_ARR[@]}"
  fi

  MODE="$([[ "${FAST}" == "1" ]] && echo fast || echo baseline)"
  echo "=== SigLIP2 multi-GPU (${MODE}) ==="
  echo "prefix=${PREFIX}  total=${WORK_TOTAL}  gpus=${GPUS}  chunk≈${CHUNK}"

  cd data_prepare
  PREV_LOG=""
  PIDS=()

  for idx in "${!GPU_ARR[@]}"; do
    GPU="${GPU_ARR[$idx]}"
    if [[ "${idx}" -gt 0 && "${WAIT_FOR_CUDA_READY}" == "1" && -n "${PREV_LOG}" ]]; then
      echo "Waiting for GPU ${GPU_ARR[$((idx - 1))]} before starting GPU ${GPU} ..."
      if ! wait_for_shard_ready "${PREV_LOG}" "${MODEL_LOAD_TIMEOUT}"; then
        echo "Abort. Restart failed shard with:"
        echo "  GPU=${GPU_ARR[$((idx - 1))]} START_INDEX=... MAX_SAMPLES=... NOHUP=1 bash scripts/run_siglip2_extract.sh"
        exit 1
      fi
      [[ "${LAUNCH_DELAY}" -gt 0 ]] && sleep "${LAUNCH_DELAY}"
    fi

    START=$(( idx * CHUNK ))
    [[ "${START}" -ge "${WORK_TOTAL}" ]] && continue
    if [[ "${idx}" -eq $((NUM_GPUS - 1)) ]]; then
      COUNT=$(( WORK_TOTAL - START ))
    else
      COUNT="${CHUNK}"
    fi

    if [[ "${FAST}" == "1" ]]; then
      LOG="${ROOT}/results/siglip2_${PREFIX}_fast_gpu${GPU}.log"
    else
      LOG="${ROOT}/results/siglip2_${PREFIX}_gpu${GPU}.log"
    fi
    [[ "${RESET_LOGS}" == "1" ]] && : > "${LOG}"

    EXTRA=(--prefix "${PREFIX}" --start_index "${START}" --max_samples "${COUNT}" \
           --seed "$((42 + START))" --log_file "${LOG}" --no_rebuild_scp)
    [[ "${REQUIRE_CUDA}" == "1" && "${FAST}" == "1" ]] && EXTRA+=(--require_cuda)
    [[ "${NO_VIDEO_PREFETCH}" == "1" && "${FAST}" == "1" ]] && EXTRA+=(--no_video_prefetch)
    [[ "${ALLOW_LOCAL}" == "1" ]] && EXTRA+=(--allow-local)

    echo "GPU ${GPU}: start=${START} count=${COUNT} log=${LOG}"
    if [[ "${FAST}" == "1" ]]; then
      CUDA_VISIBLE_DEVICES="${GPU}" nohup python feature_extractor_TVA2_siglip2_fast.py \
        --prefix "${PREFIX}" --audio_batch_size "${AUDIO_BATCH}" \
        --video_batch_size "${VIDEO_BATCH}" --video_workers "${VIDEO_WORKERS}" \
        "${EXTRA[@]}" >/dev/null 2>&1 &
    else
      CUDA_VISIBLE_DEVICES="${GPU}" nohup python feature_extractor_TVA2_siglip2.py \
        --prefix "${PREFIX}" --batch_size "${BATCH_SIZE}" \
        "${EXTRA[@]}" >/dev/null 2>&1 &
    fi
    PIDS+=("$!")
    echo "  PID ${PIDS[-1]}"
    PREV_LOG="${LOG}"
  done

  echo ""
  echo "Started ${#PIDS[@]} workers."
  for idx in "${!GPU_ARR[@]}"; do
    echo "  tail -f results/siglip2_${PREFIX}$([[ "${FAST}" == "1" ]] && echo _fast)_gpu${GPU_ARR[$idx]}.log"
  done
  print_when_done "${WORK_TOTAL}" "${SCP_TOTAL}"
  exit 0
fi

# ======================== SINGLE GPU ========================
GPU="${GPU:-0}"
if [[ "${PREFLIGHT}" == "1" ]]; then
  echo "Preflight CUDA on GPU ${GPU} ..."
  bash "${ROOT}/scripts/probe_gpus.sh" "${GPU}"
fi

if [[ "${FAST}" == "1" ]]; then
  LOG="${ROOT}/results/siglip2_${PREFIX}_fast_gpu${GPU}.log"
else
  LOG="${ROOT}/results/siglip2_${PREFIX}_gpu${GPU}.log"
fi
[[ "${RESET_LOGS}" == "1" ]] && : > "${LOG}"

EXTRA=(--prefix "${PREFIX}" --log_file "${LOG}")
[[ "${MAX_SAMPLES}" -gt 0 ]] && EXTRA+=(--max_samples "${MAX_SAMPLES}")
[[ "${START_INDEX}" -gt 0 ]] && EXTRA+=(--start_index "${START_INDEX}")
[[ "${REQUIRE_CUDA}" == "1" && "${FAST}" == "1" ]] && EXTRA+=(--require_cuda)
[[ "${NO_VIDEO_PREFETCH}" == "1" && "${FAST}" == "1" ]] && EXTRA+=(--no_video_prefetch)
[[ "${ALLOW_LOCAL}" == "1" ]] && EXTRA+=(--allow-local)
if [[ "${START_INDEX}" -gt 0 || "${MAX_SAMPLES}" -gt 0 ]]; then
  EXTRA+=(--no_rebuild_scp)
fi

MODE="$([[ "${FAST}" == "1" ]] && echo fast || echo baseline)"
echo "=== SigLIP2 single GPU (${MODE}) ==="
echo "gpu=${GPU}  prefix=${PREFIX}  scp=${SCP_TOTAL}  work=${WORK_TOTAL}  start=${START_INDEX}"
echo "log=${LOG}"

cd data_prepare
if [[ "${FAST}" == "1" ]]; then
  RUN=(python feature_extractor_TVA2_siglip2_fast.py
    --audio_batch_size "${AUDIO_BATCH}" --video_batch_size "${VIDEO_BATCH}"
    --video_workers "${VIDEO_WORKERS}" --seed "$((42 + START_INDEX))" "${EXTRA[@]}")
else
  RUN=(python feature_extractor_TVA2_siglip2.py
    --batch_size "${BATCH_SIZE}" --seed "$((42 + START_INDEX))" "${EXTRA[@]}")
fi

if [[ "${NOHUP}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" nohup "${RUN[@]}" >/dev/null 2>&1 &
  echo "Started PID $! in background."
else
  CUDA_VISIBLE_DEVICES="${GPU}" "${RUN[@]}"
fi
echo "Monitor: tail -f ${LOG}"
print_when_done "${WORK_TOTAL}" "${SCP_TOTAL}"
