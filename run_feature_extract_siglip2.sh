#!/usr/bin/env bash
# Extract TVA features with SigLIP 2 video encoder (eval_seen by default).
#
# On tars:
#   cd /local/scratch/linna/MISP/MISP_baseline/MISP-QEKS
#   bash run_feature_extract_siglip2.sh
#
# Smoke test (10 samples):
#   MAX_SAMPLES=10 bash run_feature_extract_siglip2.sh

set -euo pipefail
cd "$(dirname "$0")"

PREFIX="${PREFIX:-eval}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SIGLIP2_MODEL="${SIGLIP2_MODEL:-google/siglip2-base-patch16-224}"
BATCH_SIZE="${BATCH_SIZE:-16}"

EXTRA=()
if [[ "${MAX_SAMPLES}" -gt 0 ]]; then
  EXTRA+=(--max_samples "${MAX_SAMPLES}")
fi

echo "=== SigLIP 2 feature extraction ==="
echo "prefix=${PREFIX} model=${SIGLIP2_MODEL} batch=${BATCH_SIZE}"

# 1) raw dicts (skip if already built)
python preprocess_all.py --splits "$(python - <<PY
from paths_config import PREFIX_CONFIG
print(PREFIX_CONFIG['${PREFIX}']['data_split'])
PY
)"

# 2) features
cd data_prepare
python feature_extractor_TVA2_siglip2.py \
  --prefix "${PREFIX}" \
  --model_id "${SIGLIP2_MODEL}" \
  --batch_size "${BATCH_SIZE}" \
  "${EXTRA[@]}"

echo "Done. Video features under features/${PREFIX}/lip_siglip2/"
