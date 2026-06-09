"""
Fast TVA feature extraction: batched Qwen audio + decord/prefetch SigLIP2 video.

Compared to feature_extractor_TVA2_siglip2.py:
  - Audio: batches all pending wavs per sample (clean + SNRs) via BatchedQwenAudioEncoder
  - Video: decord decode + ThreadPoolExecutor prefetch for next clip
  - Same outputs / paths as the standard SigLIP2 extractor

Usage:
  cd data_prepare
  python feature_extractor_TVA2_siglip2_fast.py --prefix eval --allow-local
  python feature_extractor_TVA2_siglip2_fast.py --prefix train \\
      --audio_batch_size 8 --video_batch_size 32 --video_workers 4
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import re
import sys
import wave
from datetime import datetime, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))

from paths_config import (  # noqa: E402
    MATCHER_VIDEO_FEAT_DIM,
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
    setup_run_log,
    siglip2_log_path,
)

VIDEO_LIP_SUBDIR = "lip_siglip2"


def parse_args():
    p = argparse.ArgumentParser(description="Fast SigLIP2 + Qwen TVA feature extraction")
    p.add_argument("--prefix", type=str, default="eval", choices=list(PREFIX_CONFIG.keys()))
    p.add_argument("--model_id", type=str, default=SIGLIP2_MODEL_ID)
    p.add_argument("--video_batch_size", type=int, default=32, help="SigLIP2 frames per GPU batch")
    p.add_argument("--audio_batch_size", type=int, default=8, help="Qwen waveforms per GPU batch")
    p.add_argument("--video_workers", type=int, default=1, help="CPU threads for video prefetch (1 per GPU process; avoid 4×N GPUs)")
    p.add_argument(
        "--no_video_prefetch",
        action="store_true",
        help="Decode video synchronously (lower CPU load when many GPU workers run)",
    )
    p.add_argument(
        "--output_dim",
        type=int,
        default=MATCHER_VIDEO_FEAT_DIM,
        help="Project SigLIP2 to this dim (256 = XEQ-Matcher default)",
    )
    p.add_argument("--native_dim", action="store_true")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-local", action="store_true")
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument("--no_log_file", action="store_true")
    p.add_argument("--no_rebuild_scp", action="store_true")
    p.add_argument(
        "--require_cuda",
        action="store_true",
        help="Exit if CUDA is unavailable (avoid silent CPU fallback at ~70s/pair)",
    )
    return p.parse_args()


def resolve_device(require_cuda: bool, retries: int = 8, delay_sec: int = 15):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    last_err = None
    for attempt in range(1, retries + 1):
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
                torch.zeros(1, device="cuda")
                return torch.device("cuda")
            except Exception as exc:
                last_err = exc
                print(
                    f"CUDA probe attempt {attempt}/{retries} failed: {exc}",
                    file=sys.stderr,
                )
        else:
            print(
                f"CUDA not available attempt {attempt}/{retries} "
                f"(CUDA_VISIBLE_DEVICES={visible})",
                file=sys.stderr,
            )
        if attempt < retries:
            import time

            time.sleep(delay_sec)
    msg = (
        f"CUDA unavailable after {retries} attempts "
        f"(CUDA_VISIBLE_DEVICES={visible})."
    )
    if last_err is not None:
        msg += f" Last error: {last_err}"
    if require_cuda:
        print(f"ERROR: {msg}", file=sys.stderr)
        print(
            "Kill stale jobs (pkill -f feature_extractor_TVA2_siglip2), "
            "wait ~10s, restart this shard.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"WARNING: {msg} — falling back to CPU (very slow).", file=sys.stderr)
    return torch.device("cpu")


args = parse_args()
if not args.no_log_file:
    log_path = args.log_file or siglip2_log_path(f"{args.prefix}_fast")
    setup_run_log(log_path)
    print(f"Log file -> {log_path}")

_run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
print(f"=== run started {_run_ts} | start_index={args.start_index} max_samples={args.max_samples} ===")

ensure_scratch_execution(allow_local=args.allow_local)
cache_root = configure_scratch_storage()
print(f"HF/NLTK cache -> {cache_root}")

cfg = PREFIX_CONFIG[args.prefix]
scp_file = raw_scp_path(cfg["data_split"])
if not os.path.isfile(scp_file):
    print(f"ERROR: missing {scp_file}", file=sys.stderr)
    sys.exit(1)

import numpy as np
import torch
from tqdm import tqdm

from g2p.g2p_en.g2p import G2p
from qwen_audio_encoder_batched import BatchedQwenAudioEncoder, encode_pending_audio
from siglip2_video_encoder_fast import (
    FastSiglip2VideoEncoder,
    VideoPrefetcher,
    read_video_frames_fast,
)

random.seed(args.seed)
fea_save_dir = features_dir(args.prefix) + os.sep
npy_save_dir = npy_dir(args.prefix) + os.sep
noisy_wav_save_dir = noisy_wav_dir(args.prefix) + os.sep
scp_out_name = cfg["scp_name"]

snr_list = [5, 0, -5, -10]
noise_root = NOISE_ROOT
noise_list = ["Home", "Music", "TV", "Store", "WindAirCon", "WindFan", "babble_noise"]
noise_dir_map = {"Home": "GenHome", "Music": "GenMusic"}
choose_weights = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.70]

device = resolve_device(require_cuda=args.require_cuda)
_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
print(
    f"Using: {device} (CUDA_VISIBLE_DEVICES={_visible}) | "
    f"audio_batch={args.audio_batch_size} video_batch={args.video_batch_size} "
    f"prefetch={0 if args.no_video_prefetch else args.video_workers}"
)

_hf_cache = hf_cache_root()
qwen_enc = BatchedQwenAudioEncoder(
    model_id="Qwen/Qwen2-Audio-7B",
    device=device,
    max_frames=100,
    cache_dir=_hf_cache,
    batch_size=args.audio_batch_size,
)
g2p = G2p()
video_enc = FastSiglip2VideoEncoder(
    model_id=args.model_id,
    device=device,
    output_dim=None if args.native_dim else args.output_dim,
    batch_size=args.video_batch_size,
    max_frames=args.max_frames,
    cache_dir=_hf_cache,
)
prefetcher = None
if not args.no_video_prefetch and args.video_workers > 0:
    prefetcher = VideoPrefetcher(max_workers=args.video_workers, max_frames=args.max_frames)
print(f"SigLIP2 output dim: {video_enc.feat_dim}")


def read_audio(wav_path):
    with wave.open(wav_path, "rb") as wf:
        sw = wf.getsampwidth()
        audio_data = wf.readframes(wf.getnframes())
    dtype = np.int16 if sw == 2 else np.int32
    return np.frombuffer(audio_data, dtype=dtype)


def write_audio(audio_f32, path):
    i16 = (audio_f32 * 32768.0).clip(-32768, 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setparams((1, 2, 16000, len(i16), "NONE", "NONE"))
        wf.writeframes(i16.tobytes())


def audio_add_noise(clean_i16, noise_i16, snr):
    clean = clean_i16.astype(np.float32) / 32768.0
    noise = noise_i16.astype(np.float32) / 32768.0
    cp = np.mean(clean ** 2)
    if len(noise) > len(clean):
        s = random.randint(0, len(noise) - len(clean))
        noise = noise[s : s + len(clean)]
    else:
        noise = np.pad(noise, (0, len(clean) - len(noise)), "wrap")
    npow = np.mean(noise ** 2)
    if npow == 0:
        return clean
    scale = np.sqrt(cp / (10 ** (snr / 10) * npow))
    return clean + noise * scale


def audi_fea_dir(snr=None):
    sub = "wav" if snr is None else f"wav_{snr}db"
    return os.path.join(fea_save_dir, sub)


def build_data_dict(sample, anc_phn, com_phn, anc_txt_fea, com_txt_fea,
                    anc_vid_path, com_vid_path, anc_aud_path, com_aud_path,
                    anc_text, com_text, anc_lip, com_lip, anc_wav, com_wav):
    return {
        "anc_phn_list": anc_phn,
        "com_phn_list": com_phn,
        "anc_text_fea": anc_txt_fea,
        "com_text_fea": com_txt_fea,
        "anc_vide_fea_path": anc_vid_path,
        "com_vide_fea_path": com_vid_path,
        "anc_audi_fea_path": anc_aud_path,
        "com_audi_fea_path": com_aud_path,
        "type": sample["type"],
        "label": sample["label"],
        "anc_text": anc_text,
        "com_text": com_text,
        "anc_lip_path": anc_lip,
        "com_lip_path": com_lip,
        "anc_wav_path": anc_wav,
        "com_wav_path": com_wav,
        "video_encoder": "siglip2",
        "video_feat_dim": video_enc.feat_dim,
    }


def write_shuf_scp(lines, name):
    os.makedirs(data_list_dir(), exist_ok=True)
    for dest in (npy_save_dir, data_list_dir() + os.sep):
        p = os.path.join(dest, f"shuf_{name}.scp")
        with open(p, "w") as f:
            f.writelines(lines)
        print(f"Wrote {p} ({len(lines)} lines)")


def rebuild_shuf_scp(name):
    pat = re.compile(r"_-?\d+db\.npy$")
    clean = sorted(p for p in glob.glob(os.path.join(npy_save_dir, "*.npy")) if not pat.search(p))
    lines = [p + "\n" for p in clean]
    write_shuf_scp(lines, name)
    return len(lines)


def sample_complete(anc_base, com_base):
    if not os.path.exists(os.path.join(npy_save_dir, f"{anc_base}+{com_base}.npy")):
        return False
    return all(
        os.path.exists(os.path.join(npy_save_dir, f"{anc_base}+{com_base}_{snr}db.npy"))
        for snr in snr_list
    )


def encode_and_save_video(fea_path, cache_key, lip_path):
    if os.path.exists(fea_path):
        if prefetcher is not None:
            prefetcher.discard(cache_key)
        return
    if prefetcher is not None:
        frames = prefetcher.get(cache_key)
    else:
        frames = read_video_frames_fast(lip_path, max_frames=args.max_frames)
    os.makedirs(os.path.dirname(fea_path), exist_ok=True)
    np.save(fea_path, video_enc.encode(frames))


def flush_audio_jobs(jobs, wav_out=None):
    """jobs: list of (fea_path, waveform_f32); wav_out: optional {fea_path: noisy_wav_path}."""
    encoded = encode_pending_audio(qwen_enc, jobs, as_numpy=True)
    for fea_path, feat in encoded.items():
        os.makedirs(os.path.dirname(fea_path), exist_ok=True)
        np.save(fea_path, feat)
        if wav_out and fea_path in wav_out:
            write_audio(wav_out[fea_path][0], wav_out[fea_path][1])


with open(scp_file) as f:
    lines = [ln.strip() for ln in f if ln.strip()]
if args.start_index > 0:
    lines = lines[args.start_index :]
if args.max_samples > 0:
    lines = lines[: args.max_samples]

# Prefetch first sample's videos
def _schedule_pair(idx):
    if prefetcher is None or idx >= len(lines):
        return
    s = np.load(lines[idx], allow_pickle=True).item()
    vid_dir = os.path.join(fea_save_dir, VIDEO_LIP_SUBDIR)
    for side, lip in (("anc", s["anc_lip_path"]), ("com", s["com_lip_path"])):
        base = lip.lstrip("/").replace(".mp4", ".npy").replace(".m4p", ".npy")
        fea = os.path.join(vid_dir, base)
        if not os.path.exists(fea):
            prefetcher.schedule(f"{idx}:{side}", lip)


for i in range(min(args.video_workers + 1, len(lines)) if prefetcher else 0):
    _schedule_pair(i)

skipped = 0
n_gpu_audio = 0
n_gpu_video = 0
seed = args.seed

try:
    for i, line in enumerate(tqdm(lines, desc=f"siglip2-fast/{args.prefix}")):
        sample = np.load(line, allow_pickle=True).item()
        anc_wav_p, com_wav_p = sample["anc_wav_path"], sample["com_wav_path"]
        anc_base = os.path.basename(anc_wav_p).replace(".wav", "")
        com_base = os.path.basename(com_wav_p).replace(".wav", "")

        if sample_complete(anc_base, com_base):
            if prefetcher is not None:
                prefetcher.discard(f"{i}:anc")
                prefetcher.discard(f"{i}:com")
            skipped += 1
            _schedule_pair(i + args.video_workers + 1)
            continue

        _schedule_pair(i + args.video_workers + 1)

        anc_lip, com_lip = sample["anc_lip_path"], sample["com_lip_path"]
        anc_text, com_text = sample["anc_text"], sample["com_text"]
        anc_phn, anc_txt_fea = g2p(anc_text), g2p.embedding(anc_text)
        com_phn, com_txt_fea = g2p(com_text), g2p.embedding(com_text)

        vid_dir = os.path.join(fea_save_dir, VIDEO_LIP_SUBDIR)
        anc_vid = os.path.join(vid_dir, anc_lip.lstrip("/").replace(".mp4", ".npy").replace(".m4p", ".npy"))
        com_vid = os.path.join(vid_dir, com_lip.lstrip("/").replace(".mp4", ".npy").replace(".m4p", ".npy"))

        if not os.path.exists(anc_vid):
            n_gpu_video += 1
        if not os.path.exists(com_vid):
            n_gpu_video += 1
        encode_and_save_video(anc_vid, f"{i}:anc", anc_lip)
        encode_and_save_video(com_vid, f"{i}:com", com_lip)

        clean_anc = read_audio(anc_wav_p)
        clean_com = read_audio(com_wav_p)

        clean_dir = audi_fea_dir(None)
        anc_clean_fea = os.path.join(clean_dir, os.path.basename(anc_wav_p).replace(".wav", ".npy"))
        com_clean_fea = os.path.join(clean_dir, os.path.basename(com_wav_p).replace(".wav", ".npy"))

        audio_jobs = []
        if not os.path.exists(com_clean_fea):
            audio_jobs.append((com_clean_fea, clean_com.astype(np.float32) / 32768.0))
        if not os.path.exists(anc_clean_fea):
            audio_jobs.append((anc_clean_fea, clean_anc.astype(np.float32) / 32768.0))

        wav_out = {}
        for snr in snr_list:
            seed += 1
            nname = random.choices(noise_list, weights=choose_weights, k=1)[0]
            corpus = os.path.join(noise_root, noise_dir_map.get(nname, nname))
            nwavs = [w for w in os.listdir(corpus) if w.endswith(".wav")]
            noise = read_audio(os.path.join(corpus, random.choice(nwavs)))

            noisy_com = audio_add_noise(clean_com, noise, snr)
            noisy_anc = audio_add_noise(clean_anc, noise, snr)
            audi_dir = audi_fea_dir(snr)
            com_fea = os.path.join(audi_dir, os.path.basename(com_wav_p).replace(".wav", ".npy"))
            anc_fea = os.path.join(audi_dir, os.path.basename(anc_wav_p).replace(".wav", ".npy"))
            wav_dir = os.path.join(noisy_wav_save_dir, f"{args.prefix}_{snr}db")

            if not os.path.exists(com_fea):
                audio_jobs.append((com_fea, noisy_com))
                wav_out[com_fea] = (noisy_com, os.path.join(wav_dir, os.path.basename(com_wav_p)))
            if not os.path.exists(anc_fea):
                audio_jobs.append((anc_fea, noisy_anc))
                wav_out[anc_fea] = (noisy_anc, os.path.join(wav_dir, os.path.basename(anc_wav_p)))

        if audio_jobs:
            n_gpu_audio += 1
        flush_audio_jobs(audio_jobs, wav_out)

        clean_npy = os.path.join(npy_save_dir, f"{anc_base}+{com_base}.npy")
        if not os.path.exists(clean_npy):
            d = build_data_dict(
                sample, anc_phn, com_phn, anc_txt_fea, com_txt_fea,
                anc_vid, com_vid, anc_clean_fea, com_clean_fea,
                anc_text, com_text, anc_lip, com_lip, anc_wav_p, com_wav_p,
            )
            os.makedirs(os.path.dirname(clean_npy), exist_ok=True)
            np.save(clean_npy, d)

        for snr in snr_list:
            snr_npy = os.path.join(npy_save_dir, f"{anc_base}+{com_base}_{snr}db.npy")
            if os.path.exists(snr_npy):
                continue
            audi_dir = audi_fea_dir(snr)
            com_fea = os.path.join(audi_dir, os.path.basename(com_wav_p).replace(".wav", ".npy"))
            d = build_data_dict(
                sample, anc_phn, com_phn, anc_txt_fea, com_txt_fea,
                anc_vid, com_vid, anc_clean_fea, com_clean_fea,
                anc_text, com_text, anc_lip, com_lip, anc_wav_p, com_wav_p,
            )
            os.makedirs(os.path.dirname(snr_npy), exist_ok=True)
            np.save(snr_npy, d)

finally:
    if prefetcher is not None:
        prefetcher.shutdown()

if args.no_rebuild_scp:
    print(
        f"Shard done. Skipped {skipped} complete samples. "
        f"Batches with GPU audio={n_gpu_audio} video_encodes={n_gpu_video}."
    )
else:
    n = rebuild_shuf_scp(scp_out_name)
    print(
        f"Done. Skipped {skipped}. shuf has {n} entries. "
        f"GPU audio batches={n_gpu_audio} video_encodes={n_gpu_video}."
    )
