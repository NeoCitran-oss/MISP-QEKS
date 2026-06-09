#!/usr/bin/env bash
# Source before train / feature extraction.
#   source scripts/activate_env.sh
#
# Prefers local venv envs/misp-qeks; falls back to conda (mymisp on tars).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${ROOT}/envs/misp-qeks/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/envs/misp-qeks/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-mymisp}"
else
  echo "WARN: no envs/misp-qeks and no conda — using current python: $(which python)" >&2
fi

if [[ -f "${ROOT}/config/paths.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/config/paths.env"
fi

cd "${ROOT}"
echo "MISP-QEKS env active | python=$(which python)"
echo "MISP_DATA_ROOT=${MISP_DATA_ROOT:-$(python -c 'from paths_config import MISP_DATA; print(MISP_DATA)' 2>/dev/null || echo unset)}"
echo "MISP_BASELINE_ROOT=${MISP_BASELINE_ROOT:-$(python -c 'from paths_config import MISP_BASELINE; print(MISP_BASELINE)' 2>/dev/null || echo unset)}"
