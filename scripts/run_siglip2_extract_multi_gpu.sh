#!/usr/bin/env bash
# Parallel SigLIP2 feature extraction across multiple GPUs (disjoint scp shards).
#
#   # Stop any single-GPU job first:
#   pkill -f "feature_extractor_TVA2_siglip2.py --prefix train" || true
#
#   bash scripts/run_siglip2_extract_multi_gpu.sh
#   GPUS="0,4,5,6" bash scripts/run_siglip2_extract_multi_gpu.sh
#
# After ALL shards finish:
#   python scripts/rebuild_shuf_scp.py --prefix train

set -euo pipefail

BASE="/local/scratch/linna/MISP/MISP_baseline/MISP-QEKS"
PREFIX="${PREFIX:-train}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GPUS="${GPUS:-0,4,5,6}"
CONDA_ENV="${CONDA_ENV:-mymisp}"
KILL_OLD="${KILL_OLD:-1}"

cd "${BASE}"
mkdir -p results

if [[ "${KILL_OLD}" == "1" ]]; then
  if pgrep -f "feature_extractor_TVA2_siglip2.py --prefix ${PREFIX}" >/dev/null 2>&1; then
    echo "Stopping existing siglip2/${PREFIX} jobs ..."
    pkill -f "feature_extractor_TVA2_siglip2.py --prefix ${PREFIX}" || true
    sleep 2
  fi
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

SPLIT="$(python -c "from paths_config import PREFIX_CONFIG; print(PREFIX_CONFIG['${PREFIX}']['data_split'])")"
SCP="$(python -c "from paths_config import raw_scp_path, PREFIX_CONFIG; print(raw_scp_path(PREFIX_CONFIG['${PREFIX}']['data_split']))")"

if [[ ! -f "${SCP}" ]]; then
  echo "Building raw dicts for split=${SPLIT} ..."
  python preprocess_all.py --splits "${SPLIT}"
fi

TOTAL="$(wc -l < "${SCP}")"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NUM_GPUS="${#GPU_ARR[@]}"
CHUNK=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

echo "=== SigLIP2 multi-GPU extraction ==="
echo "prefix=${PREFIX}  total=${TOTAL}  gpus=${GPUS}  chunk≈${CHUNK}"

cd data_prepare
PIDS=()

for idx in "${!GPU_ARR[@]}"; do
  GPU="${GPU_ARR[$idx]}"
  START=$(( idx * CHUNK ))
  if [[ "${START}" -ge "${TOTAL}" ]]; then
    echo "Skip GPU ${GPU}: start_index ${START} >= ${TOTAL}"
    continue
  fi
  if [[ "${idx}" -eq $((NUM_GPUS - 1)) ]]; then
    COUNT=$(( TOTAL - START ))
  else
    COUNT="${CHUNK}"
  fi

  LOG="${BASE}/results/siglip2_${PREFIX}_gpu${GPU}.log"
  SEED=$(( 42 + START ))

  echo "GPU ${GPU}: start_index=${START} max_samples=${COUNT} log=${LOG}"
  CUDA_VISIBLE_DEVICES="${GPU}" nohup python feature_extractor_TVA2_siglip2.py \
    --prefix "${PREFIX}" \
    --batch_size "${BATCH_SIZE}" \
    --start_index "${START}" \
    --max_samples "${COUNT}" \
    --seed "${SEED}" \
    --log_file "${LOG}" \
    --no_rebuild_scp \
    >/dev/null 2>&1 &

  PIDS+=("$!")
  echo "  PID ${PIDS[-1]}"
done

echo ""
echo "Started ${#PIDS[@]} workers."
echo "Monitor:"
for idx in "${!GPU_ARR[@]}"; do
  GPU="${GPU_ARR[$idx]}"
  echo "  tail -f ${BASE}/results/siglip2_${PREFIX}_gpu${GPU}.log"
done
echo ""
echo "When all are done, merge scp:"
echo "  python scripts/rebuild_shuf_scp.py --prefix ${PREFIX}"
