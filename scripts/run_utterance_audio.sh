#!/usr/bin/env bash
# Phase 1a: Qwen audio embeddings only (clean + noisy SNRs).
#
#   GPU=0 PREFIX=train bash scripts/run_utterance_audio.sh
#   MAX_UTTERANCES=100 ALLOW_LOCAL=1 bash scripts/run_utterance_audio.sh  # smoke

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

GPU="${GPU:-0}"
PREFIX="${PREFIX:-train}"
AUDIO_ENCODER="${AUDIO_ENCODER:-qwen3}"
AUDIO_BATCH="${AUDIO_BATCH:-128}"
AUDIO_WORKERS="${AUDIO_WORKERS:-4}"
AUDIO_MEL_WORKERS="${AUDIO_MEL_WORKERS:-2}"
AUDIO_CHUNK="${AUDIO_CHUNK:-256}"
AUDIO_SAVE_WORKERS="${AUDIO_SAVE_WORKERS:-2}"
UTTERANCE_START="${UTTERANCE_START:-0}"
MAX_UTTERANCES="${MAX_UTTERANCES:-0}"
MEDIA_SOURCE="${MEDIA_SOURCE:-split}"
PREFLIGHT="${PREFLIGHT:-1}"
ALLOW_LOCAL="${ALLOW_LOCAL:-0}"
RESET_LOGS="${RESET_LOGS:-1}"
SAVE_NOISY_WAV="${SAVE_NOISY_WAV:-0}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p results
LOG="${ROOT}/results/audio_embed_${PREFIX}_gpu${GPU}.log"
[[ "${RESET_LOGS}" == "1" ]] && : > "${LOG}"

if [[ "${PREFLIGHT}" == "1" ]]; then
  bash "${ROOT}/scripts/probe_gpus.sh" "${GPU}"
fi

SPLIT="$(python -c "from paths_config import PREFIX_CONFIG; print(PREFIX_CONFIG['${PREFIX}']['data_split'])")"
SCP="$(python -c "from paths_config import raw_scp_path, PREFIX_CONFIG; print(raw_scp_path(PREFIX_CONFIG['${PREFIX}']['data_split']))")"
if [[ ! -f "${SCP}" ]]; then
  python preprocess_all.py --splits "${SPLIT}"
fi

EXTRA=(--prefix "${PREFIX}" --audio_only --require_cuda --no_rebuild_scp
       --log_file "${LOG}" --media_source "${MEDIA_SOURCE}"
       --audio_encoder "${AUDIO_ENCODER}"
       --audio_batch_size "${AUDIO_BATCH}" --audio_workers "${AUDIO_WORKERS}"
       --audio_mel_workers "${AUDIO_MEL_WORKERS}"
       --audio_chunk_size "${AUDIO_CHUNK}" --audio_save_workers "${AUDIO_SAVE_WORKERS}"
       --utterance_start "${UTTERANCE_START}")
[[ "${MAX_UTTERANCES}" -gt 0 ]] && EXTRA+=(--max_utterances "${MAX_UTTERANCES}")
[[ "${ALLOW_LOCAL}" == "1" ]] && EXTRA+=(--allow-local)
[[ "${SAVE_NOISY_WAV}" == "1" ]] && EXTRA+=(--save_noisy_wav)

echo "=== Qwen audio embed | gpu=${GPU} prefix=${PREFIX} encoder=${AUDIO_ENCODER} batch=${AUDIO_BATCH} workers=${AUDIO_WORKERS} log=${LOG} ==="
cd data_prepare
python feature_extractor_TVA2_siglip2_fast.py "${EXTRA[@]}"
echo "Audio extraction finished. Log: ${LOG}"
