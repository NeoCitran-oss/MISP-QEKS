#!/usr/bin/env bash
# MISP-QEKS pipeline for linna@tars.cl.uzh.ch
#
# Usage (on tars, from baseline repo root):
#   bash scripts/run_on_tars.sh eval          # preprocess + features + test (pretrained ckpt)
#   bash scripts/run_on_tars.sh train         # train from scratch (needs shuf_train.scp)
#   bash scripts/run_on_tars.sh score_fusion  # score-fusion eval
#
# Optional env vars:
#   MAX_SAMPLES=500   limit feature extraction for a quick smoke test
#   CUDA_VISIBLE_DEVICES=0

set -euo pipefail

MODE="${1:-eval}"
BASE="/local/scratch/linna/MISP/MISP_baseline/MISP-QEKS"
DATA_ROOT="/local/scratch/linna/MISP/MISP_data/MISP-QEKS"
RESULTS="${BASE}/results"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${RESULTS}/run_${MODE}_${TS}.log"

mkdir -p "${RESULTS}"
exec > >(tee -a "${LOG}") 2>&1

echo "========== MISP-QEKS pipeline =========="
echo "Mode:     ${MODE}"
echo "Host:     $(hostname)"
echo "Time:     $(date)"
echo "Base:     ${BASE}"
echo "Data:     ${DATA_ROOT}"
echo "Log:      ${LOG}"
echo "========================================"

cd "${BASE}"

# --- environment (adjust if your module names differ) ---
if [[ -f /home3/asrkws/shicheng2/bashrc_multimodal_kws ]]; then
  # shellcheck disable=SC1091
  source /home3/asrkws/shicheng2/bashrc_multimodal_kws
fi

python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())" || true
nvidia-smi || true

check_file() {
  if [[ ! -e "$1" ]]; then
    echo "MISSING: $1"
    return 1
  fi
  echo "OK: $1"
}

echo ""
echo "--- Data checks ---"
check_file "${DATA_ROOT}/data/train/wav" || true
check_file "${DATA_ROOT}/data/eval_seen/wav" || true
check_file "${DATA_ROOT}/data/noise/WindFan" || true
check_file "${DATA_ROOT}/model/lipreading/lipreading_LRW_0.8018.pt" || true
check_file "${DATA_ROOT}/train/model/epoch9.pth" || true

run_preprocess_and_features() {
  local prefix="$1"
  local split="$2"

  echo ""
  echo "--- Preprocess: ${split} ---"
  python preprocess_all.py --splits "${split}"

  echo ""
  echo "--- Feature extraction: prefix=${prefix} split=${split} ---"
  PREFIX="${prefix}" SPLIT="${split}" MAX_SAMPLES="${MAX_SAMPLES:-0}" python - <<'PY'
import os
import sys

sys.path.insert(0, "data_prepare")
os.chdir("data_prepare")

prefix = os.environ["PREFIX"]
split = os.environ["SPLIT"]
max_samples = int(os.environ.get("MAX_SAMPLES", "0"))

# Patch feature_extractor_TVA2 config at runtime
import feature_extractor_TVA2 as fe
from paths_config import PREFIX_CONFIG, raw_scp_path, features_dir, npy_dir, noisy_wav_dir, data_list_dir

cfg = PREFIX_CONFIG[prefix]
fe.prefix = prefix
fe.cfg = cfg
fe.scp_file = raw_scp_path(cfg["data_split"])
fe.fea_save_dir = features_dir(prefix) + os.sep
fe.npy_save_dir = npy_dir(prefix) + os.sep
fe.noisy_wav_save_dir = noisy_wav_dir(prefix) + os.sep
fe.scp_out_name = cfg["scp_name"]

with open(fe.scp_file) as f:
    lines = f.readlines()
if max_samples > 0:
    lines = lines[:max_samples]
    print(f"Limiting to {len(lines)} samples (MAX_SAMPLES={max_samples})")

# Re-run main loop inline by importing functions and duplicating driver
import random
import numpy as np
from tqdm import tqdm
import torch
import torchvision

shuf_scp_lines = []
seed = 42
random.seed(seed)

for line in tqdm(lines, desc=f"features/{prefix}"):
    line = line.strip()
    sample = np.load(line, allow_pickle=True).item()
    com_wav_path, anc_wav_path = sample["com_wav_path"], sample["anc_wav_path"]
    anc_lip_path, com_lip_path = sample["anc_lip_path"], sample["com_lip_path"]
    anc_text, com_text = sample["anc_text"], sample["com_text"]

    anc_phn_list, anc_text_fea = fe.TextEncoder(anc_text)
    com_phn_list, com_text_fea = fe.TextEncoder(com_text)

    vid_base_dir = os.path.join(fe.fea_save_dir, "lip")
    anc_vide_fea_path = os.path.join(vid_base_dir, anc_lip_path.lstrip("/").replace(".mp4", ".npy"))
    com_vide_fea_path = os.path.join(vid_base_dir, com_lip_path.lstrip("/").replace(".mp4", ".npy"))

    if not os.path.exists(anc_vide_fea_path):
        vid_frames, _, _ = torchvision.io.read_video(anc_lip_path, pts_unit="sec")
        anc_vide_fea = fe.VideoEncoder(vid_frames.cuda())
        os.makedirs(os.path.dirname(anc_vide_fea_path), exist_ok=True)
        np.save(anc_vide_fea_path, anc_vide_fea)

    if not os.path.exists(com_vide_fea_path):
        vid_frames, _, _ = torchvision.io.read_video(com_lip_path, pts_unit="sec")
        com_vide_fea = fe.VideoEncoder(vid_frames.cuda())
        os.makedirs(os.path.dirname(com_vide_fea_path), exist_ok=True)
        np.save(com_vide_fea_path, com_vide_fea)

    _, _, clean_com_wav = fe.read_audio(com_wav_path)
    _, _, clean_anc_wav = fe.read_audio(anc_wav_path)

    clean_audi_dir = fe.audi_fea_path_canonical(snr=None)
    com_clean_fea = os.path.join(clean_audi_dir, os.path.basename(com_wav_path).replace(".wav", ".npy"))
    anc_clean_fea = os.path.join(clean_audi_dir, os.path.basename(anc_wav_path).replace(".wav", ".npy"))
    if not os.path.exists(com_clean_fea):
        os.makedirs(os.path.dirname(com_clean_fea), exist_ok=True)
        np.save(com_clean_fea, fe.AudioEncoder(clean_com_wav / 32768.0))
    if not os.path.exists(anc_clean_fea):
        os.makedirs(os.path.dirname(anc_clean_fea), exist_ok=True)
        np.save(anc_clean_fea, fe.AudioEncoder(clean_anc_wav / 32768.0))

    anc_base = os.path.basename(anc_wav_path).replace(".wav", "")
    com_base = os.path.basename(com_wav_path).replace(".wav", "")
    clean_save_path = os.path.join(fe.npy_save_dir, f"{anc_base}+{com_base}.npy")
    if not os.path.exists(clean_save_path):
        clean_dict = fe.build_data_dict(
            sample, anc_phn_list, com_phn_list, anc_text_fea, com_text_fea,
            anc_vide_fea_path, com_vide_fea_path, anc_clean_fea, com_clean_fea,
            anc_text, com_text, anc_lip_path, com_lip_path, anc_wav_path, com_wav_path,
        )
        os.makedirs(os.path.dirname(clean_save_path), exist_ok=True)
        np.save(clean_save_path, clean_dict)
        shuf_scp_lines.append(clean_save_path + "\n")

    for snr in fe.snr_list:
        seed += 1
        noise_name = random.choices(fe.noise_list, weights=fe.choose_weights, k=1)[0]
        noise_corpus = os.path.join(fe.noise_root, fe.noise_dir_map.get(noise_name, noise_name))
        noise_wav_list = [w for w in os.listdir(noise_corpus) if w.endswith(".wav")]
        _, _, noise_wav = fe.read_audio(os.path.join(noise_corpus, random.choice(noise_wav_list)))
        noisy_com_wav = fe.audioAddNoiseScale(clean_com_wav, noise_wav, snr)
        noisy_anc_wav = fe.audioAddNoiseScale(clean_anc_wav, noise_wav, snr)
        audi_dir = fe.audi_fea_path_canonical(snr=snr)
        com_audi_fea_path = os.path.join(audi_dir, os.path.basename(com_wav_path).replace(".wav", ".npy"))
        anc_audi_fea_path = os.path.join(audi_dir, os.path.basename(anc_wav_path).replace(".wav", ".npy"))
        if not os.path.exists(com_audi_fea_path):
            os.makedirs(os.path.dirname(com_audi_fea_path), exist_ok=True)
            np.save(com_audi_fea_path, fe.AudioEncoder(noisy_com_wav))
        if not os.path.exists(anc_audi_fea_path):
            os.makedirs(os.path.dirname(anc_audi_fea_path), exist_ok=True)
            np.save(anc_audi_fea_path, fe.AudioEncoder(noisy_anc_wav))
        dict_name = f"{anc_base}+{com_base}_{snr}db.npy"
        save_path = os.path.join(fe.npy_save_dir, dict_name)
        data_dict = fe.build_data_dict(
            sample, anc_phn_list, com_phn_list, anc_text_fea, com_text_fea,
            anc_vide_fea_path, com_vide_fea_path, anc_clean_fea, com_clean_fea,
            anc_text, com_text, anc_lip_path, com_lip_path, anc_wav_path, com_wav_path,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, data_dict)

fe.write_shuf_scp(shuf_scp_lines, fe.scp_out_name)
print("Feature extraction done.")
PY
}

case "${MODE}" in
  eval)
    # Quick path: eval_seen features (set MAX_SAMPLES for smoke test)
    run_preprocess_and_features "eval" "eval_seen"

    echo ""
    echo "--- Evaluate official checkpoint (feature fusion) ---"
    bash run_test.sh

    echo ""
    echo "Results log: ${BASE}/test_epochall/test.log"
    tail -n 40 "${BASE}/test_epochall/test.log" || true
    ;;

  score_fusion)
    if [[ ! -f "${BASE}/data_list/shuf_eval_inset.scp" ]]; then
      run_preprocess_and_features "eval" "eval_seen"
    fi
    echo ""
    echo "--- Score fusion evaluation ---"
    bash run_test_score_fusion.sh
    tail -n 40 "${BASE}/test_score_fusion/test_score_fusion.log" || true
    ;;

  train)
    if [[ ! -f "${BASE}/data_list/shuf_train.scp" ]]; then
      echo "shuf_train.scp not found. Building train raw dicts + features..."
      echo "WARNING: full train feature extraction on 500k pairs takes days."
      echo "Set MAX_SAMPLES=1000 for a smoke-test train run."
      run_preprocess_and_features "train" "train"
    fi

    echo ""
    echo "--- Training ---"
    bash run_train.sh

    echo ""
    echo "--- Post-train evaluation ---"
    bash run_test.sh
    tail -n 40 "${BASE}/test_epochall/test.log" || true
    ;;

  *)
    echo "Unknown mode: ${MODE}"
    echo "Use: eval | train | score_fusion"
    exit 1
    ;;
esac

echo ""
echo "Pipeline finished. Full log: ${LOG}"
