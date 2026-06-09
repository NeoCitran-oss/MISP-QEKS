#!/usr/bin/env bash
# Quick SigLIP2 train on a partial shuf_train.scp (e.g. 50k pairs).
#
# Prereq:
#   python scripts/build_partial_train_scp.py --max_pairs 50000
#
# Usage:
#   bash run_train_siglip2_quick.sh
#   MAX_PAIRS=20000 EPOCHS=5 bash run_train_siglip2_quick.sh

set -euo pipefail
cd "$(dirname "$0")"

MAX_PAIRS="${MAX_PAIRS:-50000}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATALIST="${DATALIST:-./data_list}"
OUT_DIR="${OUT_DIR:-./train/model_siglip2_quick}"
LOG_PATH="${LOG_PATH:-./train/0_train_siglip2_quick.log}"

if [[ -f /home3/asrkws/shicheng2/bashrc_multimodal_kws ]]; then
  # shellcheck disable=SC1091
  source /home3/asrkws/shicheng2/bashrc_multimodal_kws
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-mymisp}"
fi

echo "=== Build partial train scp (max ${MAX_PAIRS}) ==="
python scripts/build_partial_train_scp.py --max_pairs "${MAX_PAIRS}"

LINES="$(wc -l < "${DATALIST}/shuf_train.scp")"
echo "shuf_train.scp has ${LINES} pairs"
if [[ "${LINES}" -lt 1000 ]]; then
  echo "ERROR: too few pairs — wait for more extraction or lower MAX_PAIRS"
  exit 1
fi

mkdir -p "$(dirname "${OUT_DIR}")" "$(dirname "${LOG_PATH}")"

echo "=== Train SigLIP2 matcher (${EPOCHS} epochs) ==="
python train.py \
  --lr 0.01 \
  --use_bmuf \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --train_snrs 5,0,-5,-10 \
  --test_snrs 5,0,-5,-10 \
  --optimizer SGD \
  --network TVA_KWS_PLCL_AVmask \
  --datalist_dir "${DATALIST}" \
  --train_csv train \
  --eval_csv eval_inset \
  --prob_addNoise 0.6 \
  --lr_half_epochs 2,3,4 \
  --out_dir "${OUT_DIR}/" \
  --log_path "${LOG_PATH}" \
  --display 40 \
  --maxlen_text 40 \
  --maxlen_vide 50 \
  --maxlen_audi 100

echo "=== Quick eval on eval_inset (last epoch) ==="
LAST=$((EPOCHS - 1))
CUDA_VISIBLE_DEVICES=0 python test.py \
  --bgn_epoch "${LAST}" \
  --end_epoch "${LAST}" \
  --batch_size 1 \
  --test_snrs 5,0,-5,-10 \
  --network TVA_KWS_PLCL_AVmask \
  --datalist_dir "${DATALIST}" \
  --eval_csv eval_inset \
  --prob_addNoise 1.0 \
  --model_path "${OUT_DIR}/" \
  --out_dir ./test_siglip2_quick/

echo "Train log: ${LOG_PATH}"
echo "Test log:  ./test_siglip2_quick/test.log"
