#!/usr/bin/env bash
# Gate on (1) eval qwen3 audio extraction and (2) train pair build, then build
# eval pairs and launch TVA training with SigLIP2 video + Qwen3 audio features.
#
#   bash scripts/run_qwen3_training.sh
#
# Markers (created by the misp-audio / misp-pairs tmux jobs):
#   results/.eval_audio_qwen3_done
#   results/.train_pairs_qwen3_done
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

source /home/linna/miniconda3/etc/profile.d/conda.sh
conda activate mymisp

TRAIN_GPUS="${TRAIN_GPUS:-0,2,3,5}"
OUT_DIR="${OUT_DIR:-./train_qwen3}"

echo "[gate] waiting for eval audio + train pairs markers..."
while [[ ! -f results/.eval_audio_qwen3_done || ! -f results/.train_pairs_qwen3_done ]]; do
  sleep 60
done
echo "[gate] both markers present, building eval pairs"

cd data_prepare
python feature_extractor_TVA2_siglip2_fast.py \
  --prefix eval --pairs_only --no_snr_pairs --audio_encoder qwen3 \
  --log_file "${ROOT}/results/pairs_eval_qwen3.log"
cd "${ROOT}"

echo "[train] launching on GPUs ${TRAIN_GPUS}"
mkdir -p "${OUT_DIR}"
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" python train.py \
  --train_snrs 3,6,9 \
  --test_snrs 3,6,9 \
  --resume \
  --out_dir "${OUT_DIR}" \
  --log_path "${OUT_DIR}/0_train.log"

echo "[train] finished"
