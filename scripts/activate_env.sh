#!/usr/bin/env bash
# Source before train / feature extraction.
#   source scripts/activate_env.sh
#
# Works under bash and zsh. Prefers envs/misp-qeks; falls back to conda mymisp.

if [[ -n "${ZSH_VERSION:-}" ]]; then
  # zsh: BASH_SOURCE is empty when sourced
  _ACTIVATE_SRC="${(%):-%x}"
elif [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _ACTIVATE_SRC="${BASH_SOURCE[0]}"
else
  _ACTIVATE_SRC="$0"
fi

ROOT="$(cd "$(dirname "${_ACTIVATE_SRC}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

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

echo "MISP-QEKS env active | python=$(which python) | cwd=${ROOT}"
if [[ -z "${MISP_DATA_ROOT:-}" ]]; then
  MISP_DATA_ROOT="$(python -c 'from paths_config import MISP_DATA; print(MISP_DATA)')"
  export MISP_DATA_ROOT
fi
if [[ -z "${MISP_BASELINE_ROOT:-}" ]]; then
  MISP_BASELINE_ROOT="$(python -c 'from paths_config import MISP_BASELINE; print(MISP_BASELINE)')"
  export MISP_BASELINE_ROOT
fi
echo "MISP_DATA_ROOT=${MISP_DATA_ROOT}"
echo "MISP_BASELINE_ROOT=${MISP_BASELINE_ROOT}"

unset _ACTIVATE_SRC
