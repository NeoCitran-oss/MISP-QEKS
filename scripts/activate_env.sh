#!/usr/bin/env bash
# Source this before running train / feature extraction on this machine.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/envs/misp-qeks/bin/activate"
if [[ -f "${ROOT}/config/paths.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/config/paths.env"
fi
cd "${ROOT}"
echo "MISP-QEKS env active | branch=$(git -C "${ROOT}" branch --show-current)"
echo "MISP_DATA_ROOT=${MISP_DATA_ROOT:-unset}"
echo "MISP_BASELINE_ROOT=${MISP_BASELINE_ROOT:-unset}"
