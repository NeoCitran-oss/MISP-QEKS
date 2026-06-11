"""
Fast TVA feature extraction — two phases:

  1. Utterances (GPU): encode each wav/mp4 once; save under features/<prefix>/
     mirroring the data/ folder layout.
  2. Pairs (CPU): build pair npy dicts that point to saved embeddings.

Usage:
  cd data_prepare
  python feature_extractor_TVA2_siglip2_fast.py --prefix eval --allow-local
  python feature_extractor_TVA2_siglip2_fast.py --prefix train --pairs_only
  python feature_extractor_TVA2_siglip2_fast.py --prefix train --utterances_only
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))

from paths_config import (  # noqa: E402
    MATCHER_VIDEO_FEAT_DIM,
    PREFIX_CONFIG,
    SIGLIP2_MODEL_ID,
    configure_scratch_storage,
    ensure_scratch_execution,
    hf_cache_root,
    raw_scp_path,
    results_dir,
    setup_run_log,
    siglip2_log_path,
)


def parse_args():
    p = argparse.ArgumentParser(description="Fast SigLIP2 + Qwen TVA (utterance-first pipeline)")
    p.add_argument("--prefix", type=str, default="eval", choices=list(PREFIX_CONFIG.keys()))
    p.add_argument("--model_id", type=str, default=SIGLIP2_MODEL_ID)
    p.add_argument("--video_batch_size", type=int, default=32)
    p.add_argument("--audio_batch_size", type=int, default=32)
    p.add_argument("--audio_workers", type=int, default=4, help="Threaded wav prefetch workers")
    p.add_argument("--audio_mel_workers", type=int, default=2, help="Threaded mel prep workers (Qwen3)")
    p.add_argument("--audio_chunk_size", type=int, default=128, help="Utterances per I/O+encode chunk")
    p.add_argument("--audio_save_workers", type=int, default=2, help="Background npy writer threads")
    p.add_argument("--no_audio_prefetch", action="store_true")
    p.add_argument("--save_noisy_wav", action="store_true", help="Also write mixed noisy wav files (slower)")
    p.add_argument(
        "--audio_encoder",
        choices=("qwen2", "qwen3"),
        default="qwen3",
        help="Audio backbone: qwen2=Qwen2-Audio-7B tower, qwen3=Qwen3-Omni AuT (faster)",
    )
    p.add_argument(
        "--audio_model_id",
        type=str,
        default=None,
        help="Override HuggingFace model id for the selected audio encoder",
    )
    p.add_argument("--video_workers", type=int, default=1)
    p.add_argument("--no_video_prefetch", action="store_true")
    p.add_argument("--output_dim", type=int, default=MATCHER_VIDEO_FEAT_DIM)
    p.add_argument("--native_dim", action="store_true")
    p.add_argument("--max_samples", type=int, default=0, help="Limit pair scp lines (phase 2)")
    p.add_argument("--start_index", type=int, default=0, help="Pair scp offset (phase 2)")
    p.add_argument("--max_utterances", type=int, default=0, help="Limit utterance files (phase 1)")
    p.add_argument("--utterance_start", type=int, default=0, help="Utterance list offset (phase 1)")
    p.add_argument(
        "--media_source",
        choices=("split", "scp"),
        default="split",
        help="split=glob all wav/mp4 under data/<split>/; scp=only paths in raw pair scp",
    )
    p.add_argument("--utterances_only", action="store_true", help="Run phase 1 only (audio + video)")
    p.add_argument("--audio_only", action="store_true", help="Phase 1: Qwen audio embeddings only")
    p.add_argument("--video_only", action="store_true", help="Phase 1: SigLIP2 video embeddings only")
    p.add_argument("--pairs_only", action="store_true", help="Run phase 2 only (embeddings must exist)")
    p.add_argument(
        "--no_snr_pairs",
        action="store_true",
        help="Skip per-SNR pair npy copies (loaders resolve noisy paths on the fly)",
    )
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-local", action="store_true")
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument("--no_log_file", action="store_true")
    p.add_argument("--no_rebuild_scp", action="store_true")
    p.add_argument("--require_cuda", action="store_true")
    return p.parse_args()


def resolve_device(require_cuda: bool, retries: int = 8, delay_sec: int = 15):
    import torch

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
                print(f"CUDA probe {attempt}/{retries} failed: {exc}", file=sys.stderr)
        if attempt < retries:
            import time

            time.sleep(delay_sec)
    msg = f"CUDA unavailable (CUDA_VISIBLE_DEVICES={visible})."
    if require_cuda:
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"WARNING: {msg} — CPU fallback.", file=sys.stderr)
    return torch.device("cpu")


def default_log_path(args) -> str:
    if args.log_file:
        return args.log_file
    gpu_tag = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if args.audio_only:
        return os.path.join(results_dir(), f"audio_embed_{args.prefix}_gpu{gpu_tag}.log")
    if args.video_only:
        return os.path.join(results_dir(), f"siglip2_embed_{args.prefix}_gpu{gpu_tag}.log")
    return siglip2_log_path(f"{args.prefix}_fast")


args = parse_args()
if args.audio_only and args.video_only:
    print("ERROR: use only one of --audio_only or --video_only", file=sys.stderr)
    sys.exit(1)

if not args.no_log_file:
    _log_path = default_log_path(args)
    setup_run_log(_log_path)
    print(f"Log file -> {_log_path}")

_run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
print(
    f"=== run started {_run_ts} | audio_only={args.audio_only} "
    f"video_only={args.video_only} utterances_only={args.utterances_only} "
    f"pairs_only={args.pairs_only} ==="
)

ensure_scratch_execution(allow_local=args.allow_local)
configure_scratch_storage()

cfg = PREFIX_CONFIG[args.prefix]
split_name = cfg["data_split"]
scp_file = raw_scp_path(split_name)
if not os.path.isfile(scp_file):
    print(f"ERROR: missing {scp_file}", file=sys.stderr)
    sys.exit(1)

with open(scp_file) as f:
    scp_lines = [ln.strip() for ln in f if ln.strip()]

import torch
from g2p.g2p_en.g2p import G2p
from pair_npy_build import run_pair_build
from qwen_audio_encoder_batched import BatchedQwenAudioEncoder, encode_audio_jobs as encode_audio_jobs_qwen2
from qwen3_audio_encoder_batched import BatchedQwen3AudioEncoder, encode_audio_jobs as encode_audio_jobs_qwen3
from audio_extract_fast import AudioPrefetcher
from siglip2_video_encoder_fast import (
    FastSiglip2VideoEncoder,
    VideoPrefetcher,
    read_video_frames_fast,
)
from utterance_extract_siglip2 import run_utterance_extraction

random.seed(args.seed)
g2p = G2p()

skip_audio = args.video_only
skip_video = args.audio_only

if args.pairs_only:
    run_utterances = False
    run_pairs = True
elif args.utterances_only or args.audio_only or args.video_only:
    run_utterances = True
    run_pairs = False
else:
    run_utterances = True
    run_pairs = True

need_audio = run_utterances and not skip_audio
need_video = run_utterances and not skip_video

prefetcher = None
audio_prefetcher = None
video_enc = None
qwen_enc = None
encode_jobs_fn = None
utter_stats = {}
pair_stats = {}

try:
    if run_utterances:
        device = resolve_device(require_cuda=args.require_cuda)
        _visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        mode = "audio" if skip_video else "video" if skip_audio else "audio+video"
        print(f"Phase 1 ({mode}) | Using: {device} (CUDA_VISIBLE_DEVICES={_visible})")
        _hf_cache = hf_cache_root()

        if need_audio:
            if args.audio_encoder == "qwen3":
                model_id = args.audio_model_id or "Qwen/Qwen3-Omni-30B-A3B-Instruct"
                qwen_enc = BatchedQwen3AudioEncoder(
                    model_id=model_id,
                    device=device,
                    max_frames=100,
                    cache_dir=_hf_cache,
                    batch_size=args.audio_batch_size,
                    mel_workers=0 if args.no_audio_prefetch else args.audio_mel_workers,
                )
                encode_jobs_fn = encode_audio_jobs_qwen3
            else:
                model_id = args.audio_model_id or "Qwen/Qwen2-Audio-7B"
                qwen_enc = BatchedQwenAudioEncoder(
                    model_id=model_id,
                    device=device,
                    max_frames=100,
                    cache_dir=_hf_cache,
                    batch_size=args.audio_batch_size,
                    use_fp16=not args.no_audio_fp16,
                )
                encode_jobs_fn = encode_audio_jobs_qwen2
            if not args.no_audio_prefetch and args.audio_workers > 0:
                audio_prefetcher = AudioPrefetcher(max_workers=args.audio_workers)
            print(
                f"Audio encoder={args.audio_encoder} model={model_id} "
                f"batch={args.audio_batch_size} chunk={args.audio_chunk_size} "
                f"workers={args.audio_workers} mel_workers={args.audio_mel_workers}"
            )

        if need_video:
            video_enc = FastSiglip2VideoEncoder(
                model_id=args.model_id,
                device=device,
                output_dim=None if args.native_dim else args.output_dim,
                batch_size=args.video_batch_size,
                max_frames=args.max_frames,
                cache_dir=_hf_cache,
            )
            if not args.no_video_prefetch and args.video_workers > 0:
                prefetcher = VideoPrefetcher(
                    max_workers=args.video_workers, max_frames=args.max_frames
                )
            print(f"SigLIP2 output dim: {video_enc.feat_dim} batch={args.video_batch_size}")

        utter_stats = run_utterance_extraction(
            prefix=args.prefix,
            split_name=split_name,
            qwen_enc=qwen_enc,
            video_enc=video_enc,
            encode_jobs_fn=encode_jobs_fn,
            read_frames_fn=read_video_frames_fast,
            max_frames=args.max_frames,
            prefetcher=prefetcher,
            audio_prefetcher=audio_prefetcher,
            seed=args.seed,
            start_index=args.utterance_start,
            max_utterances=args.max_utterances,
            skip_audio=skip_audio,
            skip_video=skip_video,
            media_source=args.media_source,
            scp_lines=scp_lines if args.media_source == "scp" else None,
            audio_chunk_size=args.audio_chunk_size,
            audio_save_workers=args.audio_save_workers,
            save_noisy_wav=args.save_noisy_wav,
            audio_encoder=args.audio_encoder,
        )

    if run_pairs:
        feat_dim = video_enc.feat_dim if video_enc is not None else args.output_dim
        print(f"Phase 2 (pairs) | assembling npy dicts from saved embeddings")
        pair_stats = run_pair_build(
            prefix=args.prefix,
            scp_lines=scp_lines,
            scp_out_name=cfg["scp_name"],
            g2p=g2p,
            video_feat_dim=feat_dim,
            start_index=args.start_index,
            max_samples=args.max_samples,
            rebuild_scp=not args.no_rebuild_scp,
            require_embeddings=True,
            audio_encoder=args.audio_encoder,
            write_snr_pairs=not args.no_snr_pairs,
        )
finally:
    if prefetcher is not None:
        prefetcher.shutdown()
    if audio_prefetcher is not None:
        audio_prefetcher.shutdown()
    if qwen_enc is not None and hasattr(qwen_enc, "shutdown"):
        qwen_enc.shutdown()

print(f"Done. utterances={utter_stats} pairs={pair_stats}")
