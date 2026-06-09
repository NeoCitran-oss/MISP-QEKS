#!/usr/bin/env bash
# Source this before running train / feature extraction on this machine.
# Works when sourced from bash or zsh.

# Resolve repo root (BASH_SOURCE is empty under zsh; use zsh %x there).
if [[ -n "${BASH_VERSION:-}" && -n "${BASH_SOURCE[0]:-}" ]]; then
  _SCRIPT="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  # shellcheck disable=SC2296
  _SCRIPT="${(%):-%x}"
else
  _SCRIPT="$0"
fi
ROOT="$(cd "$(dirname "${_SCRIPT}")/.." && pwd)"

# Python env: project venv if present, else conda (mymisp on tars).
if [[ -f "${ROOT}/envs/misp-qeks/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/envs/misp-qeks/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  if [[ -n "${ZSH_VERSION:-}" ]]; then
    eval "$(conda shell.zsh hook)"
  else
    eval "$(conda shell.bash hook)"
  fi
  conda activate "${CONDA_ENV:-mymisp}"
fi

if [[ -f "${ROOT}/config/paths.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/config/paths.env"
else
  export MISP_BASELINE_ROOT="${MISP_BASELINE_ROOT:-${ROOT}}"
  export MISP_DATA_ROOT="${MISP_DATA_ROOT:-/local/scratch/linna/MISP/MISP_data/MISP-QEKS}"
fi

cd "${ROOT}"

_BRANCH=""
if git -C "${ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  _BRANCH="$(git -C "${ROOT}" branch --show-current 2>/dev/null || true)"
fi
echo "MISP-QEKS env active | branch=${_BRANCH:-(no git)}"
echo "MISP_DATA_ROOT=${MISP_DATA_ROOT:-unset}"
echo "MISP_BASELINE_ROOT=${MISP_BASELINE_ROOT:-unset}"
