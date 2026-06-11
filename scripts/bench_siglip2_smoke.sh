#!/usr/bin/env bash
# Compare fast vs baseline SigLIP2 extraction on N eval pairs.
# Usage: bash scripts/bench_siglip2_smoke.sh [MAX_SAMPLES] [GPU]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/activate_env.sh"

PREFIX="${PREFIX:-eval}"
MAX="${1:-10}"
GPU="${2:-4}"
export CUDA_VISIBLE_DEVICES="${GPU}"

SPLIT="$(python -c "from paths_config import PREFIX_CONFIG; print(PREFIX_CONFIG['${PREFIX}']['data_split'])")"
SCP="$(python -c "from paths_config import raw_scp_path, PREFIX_CONFIG; print(raw_scp_path(PREFIX_CONFIG['${PREFIX}']['data_split']))")"
if [[ ! -f "${SCP}" ]]; then
  echo "Building raw dicts for ${SPLIT}..."
  python preprocess_all.py --splits "${SPLIT}"
fi

clear_outputs() {
  python - "${MAX}" "${PREFIX}" <<'PY'
import glob, os, re, sys
import numpy as np
from paths_config import features_dir, npy_dir, PREFIX_CONFIG, raw_scp_path

max_n = int(sys.argv[1])
prefix = sys.argv[2]
cfg = PREFIX_CONFIG[prefix]
scp_lines = open(raw_scp_path(cfg["data_split"])).read().splitlines()[:max_n]

def rm_glob(pattern):
    for p in glob.glob(pattern):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

fea = features_dir(prefix)
npy = npy_dir(prefix)
for line in scp_lines:
    s = np.load(line.strip(), allow_pickle=True).item()
    for wav in (s["anc_wav_path"], s["com_wav_path"]):
        base = os.path.basename(wav).replace(".wav", ".npy")
        rm_glob(os.path.join(fea, "wav", base))
        for snr in (5, 0, -5, -10):
            rm_glob(os.path.join(fea, f"wav_{snr}db", base))
    for lip in (s["anc_lip_path"], s["com_lip_path"]):
        base = lip.lstrip("/").replace(".mp4", ".npy").replace(".m4p", ".npy")
        rm_glob(os.path.join(fea, "lip_siglip2", base))
    anc = os.path.basename(s["anc_wav_path"]).replace(".wav", "")
    com = os.path.basename(s["com_wav_path"]).replace(".wav", "")
    rm_glob(os.path.join(npy, f"{anc}+{com}.npy"))
    for snr in (5, 0, -5, -10):
        rm_glob(os.path.join(npy, f"{anc}+{com}_{snr}db.npy"))
print(f"Cleared outputs for first {max_n} {prefix} pairs")
PY
}

run_mode() {
  local mode="$1"
  local fast="$2"
  local log="${ROOT}/results/bench_${PREFIX}_${mode}_gpu${GPU}.log"
  mkdir -p "${ROOT}/results"
  : > "${log}"
  cd "${ROOT}/data_prepare"
  local start_ts end_ts elapsed
  start_ts=$(date +%s)
  if [[ "${fast}" == "1" ]]; then
    python feature_extractor_TVA2_siglip2_fast.py \
      --prefix "${PREFIX}" --max_samples "${MAX}" --require_cuda \
      --audio_batch_size 10 --video_batch_size 32 --video_workers 2 \
      --log_file "${log}" --no_rebuild_scp --allow-local
  else
    python feature_extractor_TVA2_siglip2.py \
      --prefix "${PREFIX}" --max_samples "${MAX}" \
      --batch_size 16 --log_file "${log}" --no_rebuild_scp --allow-local
  fi
  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  echo "${elapsed}|$(grep -m1 '^Using:' "${log}" || echo 'Using: ?')"
}

echo "=== Benchmark: ${MAX} pairs, prefix=${PREFIX}, GPU=${GPU} ==="
echo "SCP: ${SCP} ($(wc -l < "${SCP}") lines)"

clear_outputs

echo ""
echo "--- Run 1/2: BASELINE (sequential) ---"
read -r T_BASE REST_BASE < <(run_mode baseline 0 | tail -1)
echo "Baseline: ${T_BASE}s | ${REST_BASE}"

clear_outputs

echo ""
echo "--- Run 2/2: FAST (batched) ---"
read -r T_FAST REST_FAST < <(run_mode fast 1 | tail -1)
echo "Fast: ${T_FAST}s | ${REST_FAST}"

python - "${MAX}" "${T_BASE}" "${T_FAST}" "${PREFIX}" "${GPU}" <<'PY'
import sys
n, t_base, t_fast, prefix, gpu = sys.argv[1:6]
n, t_base, t_fast = int(n), int(t_base), int(t_fast)
spb = lambda t: t / n
speedup = t_base / t_fast if t_fast else float("inf")
print()
print("========== BENCHMARK SUMMARY ==========")
print(f"Pairs:        {n} ({prefix})")
print(f"GPU:          {gpu}")
print(f"Baseline:     {t_base}s total  ({spb(t_base):.1f}s/pair)")
print(f"Fast:         {t_fast}s total  ({spb(t_fast):.1f}s/pair)")
print(f"Speedup:      {speedup:.2f}x faster")
print(f"Time saved:   {t_base - t_fast}s ({100 * (t_base - t_fast) / t_base:.0f}% faster)")
print("=======================================")
PY
