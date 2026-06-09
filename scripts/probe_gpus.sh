#!/usr/bin/env bash
# Quick CUDA probe — run on tars with no extraction jobs active.
#
#   bash scripts/probe_gpus.sh
#   bash scripts/probe_gpus.sh 0 1 4 5

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_env.sh"

GPUS=("$@")
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  GPUS=(0 1 2 3 4 5)
fi

echo "=== CUDA probe (one isolated python process per GPU) ==="
FAIL=0
for GPU in "${GPUS[@]}"; do
  echo ""
  echo "--- GPU ${GPU} ---"
  if ! CUDA_VISIBLE_DEVICES="${GPU}" python - <<'PY'
import os
import sys
import torch

vis = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"CUDA_VISIBLE_DEVICES={vis}")
if not torch.cuda.is_available():
    print("FAIL: torch.cuda.is_available() is False")
    sys.exit(1)
try:
    torch.cuda.set_device(0)
    x = torch.zeros(1, device="cuda")
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    print(f"OK: {name} (~{mem} GiB)")
except Exception as exc:
    print(f"FAIL: {exc}")
    sys.exit(1)
PY
  then
    FAIL=$((FAIL + 1))
  fi
done

echo ""
if [[ "${FAIL}" -gt 0 ]]; then
  echo "${FAIL} GPU(s) failed. Check: nvidia-smi"
  echo "If GPUs show processes: pkill -f feature_extractor_TVA2_siglip2; sleep 10"
  exit 1
fi
echo "All probed GPUs OK."
