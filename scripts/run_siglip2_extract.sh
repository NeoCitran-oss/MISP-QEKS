#!/usr/bin/env bash
# Background SigLIP2 feature extraction on tars (conda + preprocess + logging).
#
#   bash scripts/run_siglip2_extract.sh              # train split
#   PREFIX=eval bash scripts/run_siglip2_extract.sh  # eval_seen
#   MAX_SAMPLES=10 PREFIX=train bash scripts/run_siglip2_extract.sh  # smoke test

set -euo pipefail

BASE="/local/scratch/linna/MISP/MISP_baseline/MISP-QEKS"
PREFIX="${PREFIX:-train}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
CONDA_ENV="${CONDA_ENV:-mymisp}"

cd "${BASE}"
mkdir -p results
LOG="${BASE}/results/siglip2_${PREFIX}.log"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

echo "python: $(which python)"
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

SPLIT="$(python -c "from paths_config import PREFIX_CONFIG; print(PREFIX_CONFIG['${PREFIX}']['data_split'])")"
SCP="$(python -c "from paths_config import raw_scp_path, PREFIX_CONFIG; print(raw_scp_path(PREFIX_CONFIG['${PREFIX}']['data_split']))")"

if [[ ! -f "${SCP}" ]]; then
  echo "Building raw dicts for split=${SPLIT} ..."
  python preprocess_all.py --splits "${SPLIT}"
fi

EXTRA=()
if [[ "${MAX_SAMPLES}" -gt 0 ]]; then
  EXTRA+=(--max_samples "${MAX_SAMPLES}")
fi

cd data_prepare
nohup python feature_extractor_TVA2_siglip2.py \
  --prefix "${PREFIX}" \
  --batch_size "${BATCH_SIZE}" \
  --log_file "${LOG}" \
  "${EXTRA[@]}" >/dev/null 2>&1 &

echo "Started PID $!"
echo "Log: ${LOG}"
echo "Monitor: tail -f ${LOG}"
