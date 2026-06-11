#!/usr/bin/env bash
# CUDA / GPU diagnostics on tars (run when probe_gpus.sh fails).
#
#   bash scripts/diagnose_cuda.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "========== 1. Who am I / where =========="
echo "host=$(hostname)  user=$(whoami)  date=$(date -Is 2>/dev/null || date)"

echo ""
echo "========== 2. nvidia-smi =========="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found — are you on a GPU node?"
fi

echo ""
echo "========== 3. GPU processes (your user) =========="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>/dev/null || true
fi
echo "python/torch jobs:"
pgrep -af "python|feature_extractor|torch" 2>/dev/null || echo "(none)"

echo ""
echo "========== 4. Relevant env vars =========="
env | grep -E '^(CUDA|NVIDIA|LD_LIBRARY|PATH=)' | sort || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"

echo ""
echo "========== 5. PyTorch / CUDA build =========="
# shellcheck disable=SC1091
if [[ -f "${ROOT}/scripts/activate_env.sh" ]]; then
  source "${ROOT}/scripts/activate_env.sh" >/dev/null 2>&1 || true
fi
python - <<'PY'
import sys
print("python", sys.executable)
try:
    import torch
    print("torch", torch.__version__)
    print("torch.version.cuda", torch.version.cuda)
    print("cuda.is_available()", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count", torch.cuda.device_count())
        print("name", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch error:", e)
PY

echo ""
echo "========== 6. Minimal CUDA probe (fresh subprocess, GPU 0) =========="
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import os, sys
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
ok = torch.cuda.is_available()
print("is_available:", ok)
if not ok:
    sys.exit(1)
x = torch.zeros(1, device="cuda")
print("tensor on cuda OK")
PY
PROBE=$?
echo "probe exit code: ${PROBE}"

echo ""
echo "========== What to try =========="
cat <<'EOF'
If nvidia-smi shows your old python processes:
  pkill -u "$USER" -f feature_extractor_TVA2_siglip2
  pkill -u "$USER" -f python
  sleep 15
  bash scripts/probe_gpus.sh 0

If nvidia-smi works but probe still fails (CUDA unknown error):
  - Driver is often stuck after many kill/restart cycles on shared nodes.
  - Log out of SSH completely, wait 1–2 min, log back in, retry probe.
  - If still broken: ask cluster admin to reset GPUs or use another node.

If nvidia-smi fails entirely:
  - You may be on a login node without GPUs, or the driver is down on this host.
EOF

exit "${PROBE}"
