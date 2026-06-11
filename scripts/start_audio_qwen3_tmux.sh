#!/usr/bin/env bash
# Start Qwen3 audio extraction in tmux (does not touch video session).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmux kill-session -t misp-audio 2>/dev/null || true
pkill -u "$(whoami)" -f "feature_extractor_TVA2_siglip2_fast.py.*audio_only" 2>/dev/null || true
sleep 2
export PREFIX="${PREFIX:-train}"
export GPU="${GPU:-1}"
export AUDIO_ENCODER=qwen3
export AUDIO_BATCH="${AUDIO_BATCH:-128}"
export AUDIO_MEL_WORKERS="${AUDIO_MEL_WORKERS:-2}"
export MEDIA_SOURCE="${MEDIA_SOURCE:-split}"
export PREFLIGHT="${PREFLIGHT:-1}"
export RESET_LOGS="${RESET_LOGS:-0}"
tmux new-session -d -s misp-audio \
  "cd ${ROOT} && PREFIX=${PREFIX} GPU=${GPU} AUDIO_ENCODER=qwen3 AUDIO_BATCH=${AUDIO_BATCH:-128} AUDIO_MEL_WORKERS=${AUDIO_MEL_WORKERS:-2} MEDIA_SOURCE=${MEDIA_SOURCE:-split} RESET_LOGS=${RESET_LOGS:-0} PREFLIGHT=${PREFLIGHT:-1} bash scripts/run_utterance_audio.sh; echo; echo '[tmux] audio done'; read"
echo "Started misp-audio on GPU ${GPU} (encoder=qwen3)"
echo "  tmux attach -t misp-audio"
echo "  log: results/audio_embed_${PREFIX}_gpu${GPU}.log"
