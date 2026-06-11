#!/usr/bin/env bash
# Start audio + SigLIP2 utterance extraction in two tmux sessions on different GPUs.
#
#   bash scripts/tmux_dual_extract.sh
#   PREFIX=train AUDIO_GPU=0 VIDEO_GPU=4 bash scripts/tmux_dual_extract.sh
#
# Attach:
#   tmux attach -t misp-audio
#   tmux attach -t misp-siglip2
#
# After both finish, build pair npy (CPU):
#   PREFIX=train PAIRS_ONLY=1 bash scripts/run_siglip2_extract.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PREFIX="${PREFIX:-train}"
AUDIO_GPU="${AUDIO_GPU:-0}"
VIDEO_GPU="${VIDEO_GPU:-4}"
SESSION_AUDIO="${SESSION_AUDIO:-misp-audio}"
SESSION_SIGLIP2="${SESSION_SIGLIP2:-misp-siglip2}"
KILL_OLD="${KILL_OLD:-1}"
MAX_UTTERANCES="${MAX_UTTERANCES:-0}"
UTTERANCE_START="${UTTERANCE_START:-0}"
MEDIA_SOURCE="${MEDIA_SOURCE:-split}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found" >&2
  exit 1
fi

chmod +x "${ROOT}/scripts/run_utterance_audio.sh" "${ROOT}/scripts/run_utterance_siglip2.sh"

if [[ "${KILL_OLD}" == "1" ]]; then
  tmux kill-session -t "${SESSION_AUDIO}" 2>/dev/null || true
  tmux kill-session -t "${SESSION_SIGLIP2}" 2>/dev/null || true
  pkill -u "$(whoami)" -f "feature_extractor_TVA2_siglip2_fast.py" 2>/dev/null || true
  sleep 3
fi

common_env="cd ${ROOT} && export PREFIX=${PREFIX} MAX_UTTERANCES=${MAX_UTTERANCES} UTTERANCE_START=${UTTERANCE_START} MEDIA_SOURCE=${MEDIA_SOURCE} PREFLIGHT=1 RESET_LOGS=1"

audio_cmd="${common_env} && GPU=${AUDIO_GPU} bash scripts/run_utterance_audio.sh"
video_cmd="${common_env} && GPU=${VIDEO_GPU} bash scripts/run_utterance_siglip2.sh"

tmux new-session -d -s "${SESSION_AUDIO}" "bash -lc '${audio_cmd}; echo; echo [tmux] audio session exited — press Enter; read'"
tmux new-session -d -s "${SESSION_SIGLIP2}" "bash -lc '${video_cmd}; echo; echo [tmux] siglip2 session exited — press Enter; read'"

echo "Started tmux sessions on tars:"
echo "  ${SESSION_AUDIO}  GPU ${AUDIO_GPU}  log: results/audio_embed_${PREFIX}_gpu${AUDIO_GPU}.log"
echo "  ${SESSION_SIGLIP2}  GPU ${VIDEO_GPU}  log: results/siglip2_embed_${PREFIX}_gpu${VIDEO_GPU}.log"
echo ""
echo "  tmux attach -t ${SESSION_AUDIO}"
echo "  tmux attach -t ${SESSION_SIGLIP2}"
echo "  tmux ls"
echo ""
echo "When both done:"
echo "  PREFIX=${PREFIX} PAIRS_ONLY=1 bash scripts/run_siglip2_extract.sh"
