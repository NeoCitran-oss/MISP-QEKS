"""
Feature extraction with Google SigLIP 2 video encoder + Qwen2-Audio + G2P.

Same pipeline as feature_extractor_TVA2.py, but lip/video features are stored
under features/<prefix>/lip_siglip2/ (does not overwrite CNN lip features).

Run on tars only (not your laptop) — caches and outputs go to /local/scratch.

Usage (on tars):
  cd data_prepare
  python feature_extractor_TVA2_siglip2.py --prefix eval

Requires:
  pip install "transformers>=4.49.0" pillow
"""
import argparse
import glob
import os
import random
import re
import sys
import wave

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from paths_config import (  # noqa: E402
    NOISE_ROOT,
    PREFIX_CONFIG,
    SIGLIP2_MODEL_ID,
    configure_scratch_storage,
    data_list_dir,
    ensure_scratch_execution,
    features_dir,
    hf_cache_root,
    noisy_wav_dir,
    npy_dir,
    raw_scp_path,
)
from siglip2_video_encoder import MATCHER_VIDEO_FEAT_DIM  # noqa: E402

VIDEO_LIP_SUBDIR = "lip_siglip2"


def parse_args():
    parser = argparse.ArgumentParser(description="TVA feature extraction with SigLIP 2 video encoder")
    parser.add_argument("--prefix", type=str, default="eval", choices=list(PREFIX_CONFIG.keys()))
    parser.add_argument("--model_id", type=str, default=SIGLIP2_MODEL_ID)
    parser.add_argument("--batch_size", type=int, default=16, help="SigLIP 2 frame batch size")
    parser.add_argument(
        "--output_dim",
        type=int,
        default=MATCHER_VIDEO_FEAT_DIM,
        help="Project SigLIP 2 embeddings to this dim (256 matches XEQ-Matcher Vide_Proj)",
    )
    parser.add_argument(
        "--native_dim",
        action="store_true",
        help="Save native SigLIP 2 dim (no linear projection); requires retraining Vide_Proj",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="Limit samples (0 = all)")
    parser.add_argument("--start_index", type=int, default=0, help="Resume: skip first N lines in raw scp")
    parser.add_argument("--max_frames", type=int, default=50, help="Max video frames per clip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-local",
        action="store_true",
        help="Allow running off tars (downloads HF models to local disk — not recommended)",
    )
    return parser.parse_args()


args = parse_args()
ensure_scratch_execution(allow_local=args.allow_local)
cache_root = configure_scratch_storage()
print(f"HF/NLTK cache -> {cache_root}")

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from g2p.g2p_en.g2p import G2p
from qwen_audio_encoder import QwenAudioEncoder
from siglip2_video_encoder import Siglip2VideoEncoder

random.seed(args.seed)

cfg = PREFIX_CONFIG[args.prefix]
scp_file = raw_scp_path(cfg["data_split"])
fea_save_dir = features_dir(args.prefix) + os.sep
npy_save_dir = npy_dir(args.prefix) + os.sep
noisy_wav_save_dir = noisy_wav_dir(args.prefix) + os.sep
scp_out_name = cfg["scp_name"]

snr_list = [5, 0, -5, -10]
noise_root = NOISE_ROOT
noise_list = ["Home", "Music", "TV", "Store", "WindAirCon", "WindFan", "babble_noise"]
noise_dir_map = {"Home": "GenHome", "Music": "GenMusic"}
choose_weights = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.70]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
print("SigLIP 2 model:", args.model_id)
print("Video features ->", VIDEO_LIP_SUBDIR)

_hf_cache = hf_cache_root()
qwen_enc = QwenAudioEncoder(
    model_id="Qwen/Qwen2-Audio-7B",
    device=device,
    max_frames=100,
    cache_dir=_hf_cache,
)
g2p = G2p()
video_enc = Siglip2VideoEncoder(
    model_id=args.model_id,
    device=device,
    output_dim=None if args.native_dim else args.output_dim,
    batch_size=args.batch_size,
    max_frames=args.max_frames,
    cache_dir=_hf_cache,
)
print(f"SigLIP 2 output dim: {video_enc.feat_dim} (native={video_enc.native_feat_dim})")


def read_audio(wav_path):
    with wave.open(wav_path, "rb") as wf:
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
    dtype = np.int16 if sample_width == 2 else np.int32
    return sample_width, frame_rate, np.frombuffer(audio_data, dtype=dtype)


def write_audio(audio_data, audio_name):
    audio_data_int16 = (audio_data * 32768.0).clip(-32768, 32767).astype(np.int16)
    wave_file = wave.open(audio_name, "wb")
    wave_file.setparams((1, 2, 16000, len(audio_data_int16), "NONE", "NONE"))
    wave_file.writeframes(audio_data_int16.tobytes())
    wave_file.close()


def AudioEncoder(audio, encoder=qwen_enc):
    return encoder.encode(audio)


def TextEncoder(text, encoder=g2p):
    return encoder(text), torch.from_numpy(encoder.embedding(text)).numpy()


def VideoEncoder(frames: torch.Tensor, encoder=video_enc) -> np.ndarray:
    return encoder.encode(frames)


def audioAddNoiseScale(clean_wav, noise_wav, snr):
    clean_wav = np.array(clean_wav, dtype=np.float32) / 32768.0
    noise_wav = np.array(noise_wav, dtype=np.float32) / 32768.0
    clean_power = np.mean(clean_wav ** 2)
    if len(noise_wav) > len(clean_wav):
        start_idx = random.randint(0, len(noise_wav) - len(clean_wav))
        noise_wav = noise_wav[start_idx : start_idx + len(clean_wav)]
    else:
        noise_wav = np.pad(noise_wav, (0, len(clean_wav) - len(noise_wav)), "wrap")
    noise_power = np.mean(noise_wav ** 2)
    if noise_power == 0:
        return clean_wav
    scaling_factor = np.sqrt(clean_power / (10 ** (snr / 10) * noise_power))
    return clean_wav + (noise_wav * scaling_factor)


def audi_fea_path_canonical(snr=None):
    sub = "wav" if snr is None else f"wav_{snr}db"
    return os.path.join(fea_save_dir, sub)


def build_data_dict(
    sample,
    anc_phn_list,
    com_phn_list,
    anc_text_fea,
    com_text_fea,
    anc_vide_fea_path,
    com_vide_fea_path,
    anc_audi_fea_path,
    com_audi_fea_path,
    anc_text,
    com_text,
    anc_lip_path,
    com_lip_path,
    anc_wav_path,
    com_wav_path,
):
    return {
        "anc_phn_list": anc_phn_list,
        "com_phn_list": com_phn_list,
        "anc_text_fea": anc_text_fea,
        "com_text_fea": com_text_fea,
        "anc_vide_fea_path": anc_vide_fea_path,
        "com_vide_fea_path": com_vide_fea_path,
        "anc_audi_fea_path": anc_audi_fea_path,
        "com_audi_fea_path": com_audi_fea_path,
        "type": sample["type"],
        "label": sample["label"],
        "anc_text": anc_text,
        "com_text": com_text,
        "anc_lip_path": anc_lip_path,
        "com_lip_path": com_lip_path,
        "anc_wav_path": anc_wav_path,
        "com_wav_path": com_wav_path,
        "video_encoder": "siglip2",
        "video_feat_dim": video_enc.feat_dim,
    }


def write_shuf_scp(lines, scp_name):
    os.makedirs(data_list_dir(), exist_ok=True)
    for dest in (npy_save_dir, data_list_dir() + os.sep):
        shuf_path = os.path.join(dest, f"shuf_{scp_name}.scp")
        with open(shuf_path, "w") as f:
            f.writelines(lines)
        print(f"Wrote {shuf_path} ({len(lines)} lines)")


def rebuild_shuf_scp_from_disk(scp_name):
    """Rebuild shuf list from all clean npy dicts (safe after resume)."""
    pattern = re.compile(r"_-?\d+db\.npy$")
    clean_npys = sorted(
        p for p in glob.glob(os.path.join(npy_save_dir, "*.npy")) if not pattern.search(p)
    )
    lines = [p + "\n" for p in clean_npys]
    write_shuf_scp(lines, scp_name)
    return len(lines)


def sample_is_complete(anc_base, com_base):
    clean_path = os.path.join(npy_save_dir, f"{anc_base}+{com_base}.npy")
    if not os.path.exists(clean_path):
        return False
    for snr in snr_list:
        snr_path = os.path.join(npy_save_dir, f"{anc_base}+{com_base}_{snr}db.npy")
        if not os.path.exists(snr_path):
            return False
    return True


def read_video_frames(path: str) -> torch.Tensor:
    frames, _, _ = torchvision.io.read_video(path, pts_unit="sec")
    return frames


with open(scp_file) as f:
    lines = f.readlines()
if args.start_index > 0:
    lines = lines[args.start_index :]
    print(f"Resuming from index {args.start_index} ({len(lines)} samples remaining)")
if args.max_samples > 0:
    lines = lines[: args.max_samples]
    print(f"Limited to {len(lines)} samples")

seed = args.seed
skipped_complete = 0

for line in tqdm(lines, desc=f"siglip2/{args.prefix}"):
    line = line.strip()
    sample = np.load(line, allow_pickle=True).item()

    com_wav_path, anc_wav_path = sample["com_wav_path"], sample["anc_wav_path"]
    anc_base = os.path.basename(anc_wav_path).replace(".wav", "")
    com_base = os.path.basename(com_wav_path).replace(".wav", "")
    if sample_is_complete(anc_base, com_base):
        skipped_complete += 1
        continue

    anc_lip_path, com_lip_path = sample["anc_lip_path"], sample["com_lip_path"]
    anc_text, com_text = sample["anc_text"], sample["com_text"]

    anc_phn_list, anc_text_fea = TextEncoder(anc_text)
    com_phn_list, com_text_fea = TextEncoder(com_text)

    vid_base_dir = os.path.join(fea_save_dir, VIDEO_LIP_SUBDIR)
    anc_vide_fea_path = os.path.join(
        vid_base_dir, anc_lip_path.lstrip("/").replace(".mp4", ".npy").replace(".m4p", ".npy")
    )
    com_vide_fea_path = os.path.join(
        vid_base_dir, com_lip_path.lstrip("/").replace(".mp4", ".npy").replace(".m4p", ".npy")
    )

    if not os.path.exists(anc_vide_fea_path):
        anc_frames = read_video_frames(anc_lip_path)
        os.makedirs(os.path.dirname(anc_vide_fea_path), exist_ok=True)
        np.save(anc_vide_fea_path, VideoEncoder(anc_frames))

    if not os.path.exists(com_vide_fea_path):
        com_frames = read_video_frames(com_lip_path)
        os.makedirs(os.path.dirname(com_vide_fea_path), exist_ok=True)
        np.save(com_vide_fea_path, VideoEncoder(com_frames))

    _, _, clean_com_wav = read_audio(com_wav_path)
    _, _, clean_anc_wav = read_audio(anc_wav_path)

    clean_audi_dir = audi_fea_path_canonical(snr=None)
    com_clean_fea = os.path.join(
        clean_audi_dir, os.path.basename(com_wav_path).replace(".wav", ".npy")
    )
    anc_clean_fea = os.path.join(
        clean_audi_dir, os.path.basename(anc_wav_path).replace(".wav", ".npy")
    )
    if not os.path.exists(com_clean_fea):
        os.makedirs(os.path.dirname(com_clean_fea), exist_ok=True)
        np.save(com_clean_fea, AudioEncoder(clean_com_wav / 32768.0))
    if not os.path.exists(anc_clean_fea):
        os.makedirs(os.path.dirname(anc_clean_fea), exist_ok=True)
        np.save(anc_clean_fea, AudioEncoder(clean_anc_wav / 32768.0))

    clean_save_path = os.path.join(npy_save_dir, f"{anc_base}+{com_base}.npy")
    if not os.path.exists(clean_save_path):
        clean_dict = build_data_dict(
            sample,
            anc_phn_list,
            com_phn_list,
            anc_text_fea,
            com_text_fea,
            anc_vide_fea_path,
            com_vide_fea_path,
            anc_clean_fea,
            com_clean_fea,
            anc_text,
            com_text,
            anc_lip_path,
            com_lip_path,
            anc_wav_path,
            com_wav_path,
        )
        os.makedirs(os.path.dirname(clean_save_path), exist_ok=True)
        np.save(clean_save_path, clean_dict)

    for snr in snr_list:
        seed += 1
        noise_name = random.choices(noise_list, weights=choose_weights, k=1)[0]
        noise_corpus = os.path.join(noise_root, noise_dir_map.get(noise_name, noise_name))
        noise_wav_list = [w for w in os.listdir(noise_corpus) if w.endswith(".wav")]
        if not noise_wav_list:
            raise FileNotFoundError(f"No .wav files in {noise_corpus}")
        _, _, noise_wav = read_audio(os.path.join(noise_corpus, random.choice(noise_wav_list)))

        noisy_com_wav = audioAddNoiseScale(clean_com_wav, noise_wav, snr)
        noisy_anc_wav = audioAddNoiseScale(clean_anc_wav, noise_wav, snr)

        wav_save_dir = os.path.join(noisy_wav_save_dir, f"{args.prefix}_{snr}db")
        audi_dir = audi_fea_path_canonical(snr=snr)
        com_audi_fea_path = os.path.join(
            audi_dir, os.path.basename(com_wav_path).replace(".wav", ".npy")
        )
        anc_audi_fea_path = os.path.join(
            audi_dir, os.path.basename(anc_wav_path).replace(".wav", ".npy")
        )

        if not os.path.exists(com_audi_fea_path):
            os.makedirs(os.path.dirname(com_audi_fea_path), exist_ok=True)
            np.save(com_audi_fea_path, AudioEncoder(noisy_com_wav))
            com_noisy_path = os.path.join(wav_save_dir, os.path.basename(com_wav_path))
            os.makedirs(os.path.dirname(com_noisy_path), exist_ok=True)
            write_audio(noisy_com_wav, com_noisy_path)

        if not os.path.exists(anc_audi_fea_path):
            os.makedirs(os.path.dirname(anc_audi_fea_path), exist_ok=True)
            np.save(anc_audi_fea_path, AudioEncoder(noisy_anc_wav))
            anc_noisy_path = os.path.join(wav_save_dir, os.path.basename(anc_wav_path))
            os.makedirs(os.path.dirname(anc_noisy_path), exist_ok=True)
            write_audio(noisy_anc_wav, anc_noisy_path)

        dict_name = f"{anc_base}+{com_base}_{snr}db.npy"
        save_path = os.path.join(npy_save_dir, dict_name)
        if not os.path.exists(save_path):
            data_dict = build_data_dict(
                sample,
                anc_phn_list,
                com_phn_list,
                anc_text_fea,
                com_text_fea,
                anc_vide_fea_path,
                com_vide_fea_path,
                anc_clean_fea,
                com_clean_fea,
                anc_text,
                com_text,
                anc_lip_path,
                com_lip_path,
                anc_wav_path,
                com_wav_path,
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, data_dict)

n_clean = rebuild_shuf_scp_from_disk(scp_out_name)
print(f"Done. Skipped {skipped_complete} already-complete samples. shuf has {n_clean} entries.")
