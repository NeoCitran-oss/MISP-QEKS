#!/usr/bin/env bash
# Phase 1b: SigLIP2 video embeddings only.
#
#   GPU=4 PREFIX=train bash scripts/run_utterance_siglip2.sh
#   MAX_UTTERANCES=100 ALLOW_LOCAL=1 bash scripts/run_utterance_siglip2.sh  # smoke

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f /home/linna/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/linna/miniconda3/etc/profile.d/conda.sh
  conda activate mymisp
else
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/activate_env.sh"
fi

GPU="${GPU:-4}"
PREFIX="${PREFIX:-train}"
VIDEO_BATCH="${VIDEO_BATCH:-32}"
VIDEO_WORKERS="${VIDEO_WORKERS:-2}"
NO_VIDEO_PREFETCH="${NO_VIDEO_PREFETCH:-0}"
UTTERANCE_START="${UTTERANCE_START:-0}"
MAX_UTTERANCES="${MAX_UTTERANCES:-0}"
MEDIA_SOURCE="${MEDIA_SOURCE:-split}"
PREFLIGHT="${PREFLIGHT:-1}"
ALLOW_LOCAL="${ALLOW_LOCAL:-0}"
RESET_LOGS="${RESET_LOGS:-1}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p results
LOG="${ROOT}/results/siglip2_embed_${PREFIX}_gpu${GPU}.log"
[[ "${RESET_LOGS}" == "1" ]] && : > "${LOG}"

if [[ "${PREFLIGHT}" == "1" ]]; then
  bash "${ROOT}/scripts/probe_gpus.sh" "${GPU}"
fi

SPLIT="$(python -c "from paths_config import PREFIX_CONFIG; print(PREFIX_CONFIG['${PREFIX}']['data_split'])")"
SCP="$(python -c "from paths_config import raw_scp_path, PREFIX_CONFIG; print(raw_scp_path(PREFIX_CONFIG['${PREFIX}']['data_split']))")"
if [[ ! -f "${SCP}" ]]; then
  python preprocess_all.py --splits "${SPLIT}"
fi

EXTRA=(--prefix "${PREFIX}" --video_only --require_cuda --no_rebuild_scp
       --log_file "${LOG}" --media_source "${MEDIA_SOURCE}"
       --video_batch_size "${VIDEO_BATCH}" --video_workers "${VIDEO_WORKERS}"
       --utterance_start "${UTTERANCE_START}")
[[ "${MAX_UTTERANCES}" -gt 0 ]] && EXTRA+=(--max_utterances "${MAX_UTTERANCES}")
[[ "${ALLOW_LOCAL}" == "1" ]] && EXTRA+=(--allow-local)
[[ "${NO_VIDEO_PREFETCH}" == "1" ]] && EXTRA+=(--no_video_prefetch)

echo "=== SigLIP2 video embed | gpu=${GPU} prefix=${PREFIX} log=${LOG} ==="
cd data_prepare
python feature_extractor_TVA2_siglip2_fast.py "${EXTRA[@]}"
echo "SigLIP2 extraction finished. Log: ${LOG}"
