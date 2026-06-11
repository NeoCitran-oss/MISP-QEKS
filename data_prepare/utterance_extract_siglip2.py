"""
Phase 1: extract per-utterance audio/video embeddings (GPU).

Each source wav/mp4 is encoded once; outputs mirror the data/ tree under
features/<prefix>/{wav,wav_*db,lip_siglip2}/.

Run before pair_npy_build.py or via feature_extractor_TVA2_siglip2_fast.py.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import numpy as np
from tqdm import tqdm

from audio_utils import audio_add_noise, noise_rng_for_utterance, read_audio, write_audio
from audio_extract_fast import (
    AsyncNpySaver,
    AudioPrefetcher,
    NoiseCorpusPool,
    prefetch_ahead,
    read_audio_f32,
)
from embed_paths import (
    SNR_LIST,
    audio_embed_path,
    audio_embedding_exists,
    list_split_media,
    video_embed_path,
)
from paths_config import NOISE_ROOT, noisy_wav_dir


def extract_videos(
    video_paths: Sequence[str],
    prefix: str,
    video_enc,
    *,
    read_frames_fn,
    max_frames: int = 50,
    prefetcher=None,
    prefetch_ahead: int = 2,
) -> int:
    n_enc = 0
    pending = [p for p in video_paths if not os.path.exists(video_embed_path(prefix, p))]
    if not pending:
        return 0

    def schedule_idx(idx: int) -> None:
        if prefetcher is not None and idx < len(pending):
            prefetcher.schedule(f"vid:{idx}", pending[idx])

    ahead = prefetch_ahead if prefetcher is not None else 0
    for j in range(min(ahead + 1, len(pending))):
        schedule_idx(j)

    for i, lip_path in enumerate(tqdm(pending, desc=f"utterances/video/{prefix}")):
        out_path = video_embed_path(prefix, lip_path)
        if prefetcher is not None:
            frames = prefetcher.get(f"vid:{i}")
            schedule_idx(i + ahead + 1)
        else:
            frames = read_frames_fn(lip_path, max_frames=max_frames)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, video_enc.encode(frames))
        n_enc += 1
    return n_enc


def extract_clean_audio_batch(
    wav_paths: Sequence[str],
    prefix: str,
    qwen_enc,
    encode_jobs_fn,
    *,
    wav_chunk: int = 128,
    audio_prefetcher: Optional[AudioPrefetcher] = None,
    save_workers: int = 2,
    prefetch_ahead_n: int = 8,
    audio_encoder: str = "qwen2",
) -> int:
    pending_paths = [
        wav_path
        for wav_path in wav_paths
        if not audio_embedding_exists(prefix, wav_path, snr=None, encoder=audio_encoder)
    ]
    if not pending_paths:
        return 0

    saver = AsyncNpySaver(max_workers=save_workers)
    total = 0
    ahead = prefetch_ahead_n if audio_prefetcher is not None else 0
    prefetch_ahead(audio_prefetcher, [str(i) for i in range(len(pending_paths))], pending_paths, 0, ahead + 1)

    try:
        for start in tqdm(
            range(0, len(pending_paths), wav_chunk),
            desc=f"utterances/audio_clean/{prefix}",
        ):
            chunk_paths = pending_paths[start : start + wav_chunk]
            jobs = []
            for i, wav_path in enumerate(chunk_paths):
                key = str(start + i)
                if audio_prefetcher is not None:
                    clean = audio_prefetcher.get(key)
                    nxt = start + i + ahead + 1
                    if nxt < len(pending_paths):
                        audio_prefetcher.schedule(str(nxt), pending_paths[nxt])
                else:
                    clean = read_audio_f32(wav_path)
                jobs.append((audio_embed_path(prefix, wav_path, snr=None, encoder=audio_encoder), clean))
            encoded = encode_jobs_fn(qwen_enc, jobs, as_numpy=True)
            for out_path, feat in encoded.items():
                saver.save(out_path, feat)
            total += len(encoded)
    finally:
        saver.shutdown()
    return total


def extract_noisy_audio_batch(
    wav_paths: Sequence[str],
    prefix: str,
    qwen_enc,
    encode_jobs_fn,
    *,
    noise_root: str = NOISE_ROOT,
    seed: int = 42,
    noisy_wav_root: Optional[str] = None,
    save_noisy_wav: bool = False,
    wav_chunk: int = 64,
    audio_prefetcher: Optional[AudioPrefetcher] = None,
    save_workers: int = 2,
    prefetch_ahead_n: int = 8,
    audio_encoder: str = "qwen2",
) -> int:
    pending_paths = [
        wav_path
        for wav_path in wav_paths
        if any(
            not audio_embedding_exists(prefix, wav_path, snr=snr, encoder=audio_encoder)
            for snr in SNR_LIST
        )
    ]
    if not pending_paths:
        return 0

    if noisy_wav_root is None:
        noisy_wav_root = noisy_wav_dir(prefix)

    noise_pool = NoiseCorpusPool(noise_root)
    saver = AsyncNpySaver(max_workers=save_workers)
    total = 0
    ahead = prefetch_ahead_n if audio_prefetcher is not None else 0
    prefetch_ahead(audio_prefetcher, [str(i) for i in range(len(pending_paths))], pending_paths, 0, ahead + 1)

    try:
        for start in tqdm(
            range(0, len(pending_paths), wav_chunk),
            desc=f"utterances/audio_noisy/{prefix}",
        ):
            chunk_paths = pending_paths[start : start + wav_chunk]
            jobs = []
            wav_out = {}
            for i, wav_path in enumerate(chunk_paths):
                key = str(start + i)
                if audio_prefetcher is not None:
                    clean_f32 = audio_prefetcher.get(key)
                    clean_i16 = (clean_f32 * 32768.0).astype(np.int16)
                    nxt = start + i + ahead + 1
                    if nxt < len(pending_paths):
                        audio_prefetcher.schedule(str(nxt), pending_paths[nxt])
                else:
                    clean_i16 = read_audio(wav_path)

                for snr in SNR_LIST:
                    out_path = audio_embed_path(prefix, wav_path, snr=snr, encoder=audio_encoder)
                    if audio_embedding_exists(prefix, wav_path, snr=snr, encoder=audio_encoder):
                        continue
                    rng = noise_rng_for_utterance(wav_path, snr, seed)
                    noise_i16 = read_audio(noise_pool.pick(rng))
                    noisy = audio_add_noise(clean_i16, noise_i16, snr, rng=rng)
                    jobs.append((out_path, noisy))
                    if save_noisy_wav:
                        rel = os.path.basename(wav_path)
                        wav_out[out_path] = (
                            noisy,
                            os.path.join(noisy_wav_root, f"{prefix}_{snr}db", rel),
                        )
            if not jobs:
                continue
            encoded = encode_jobs_fn(qwen_enc, jobs, as_numpy=True)
            for out_path, feat in encoded.items():
                saver.save(out_path, feat)
                if save_noisy_wav and out_path in wav_out:
                    noisy, wav_path_out = wav_out[out_path]
                    write_audio(noisy, wav_path_out)
            total += len(encoded)
    finally:
        saver.shutdown()
    return total


def run_utterance_extraction(
    *,
    prefix: str,
    split_name: str,
    qwen_enc,
    video_enc,
    encode_jobs_fn,
    read_frames_fn,
    max_frames: int = 50,
    prefetcher=None,
    audio_prefetcher: Optional[AudioPrefetcher] = None,
    seed: int = 42,
    start_index: int = 0,
    max_utterances: int = 0,
    skip_video: bool = False,
    skip_audio: bool = False,
    media_source: str = "split",
    scp_lines: Optional[Iterable[str]] = None,
    audio_chunk_size: int = 128,
    audio_save_workers: int = 2,
    save_noisy_wav: bool = False,
    audio_encoder: str = "qwen2",
) -> dict:
    if media_source == "scp":
        if scp_lines is None:
            raise ValueError("scp_lines required when media_source='scp'")
        from embed_paths import collect_utterances_from_scp

        wav_set, vid_set = collect_utterances_from_scp(scp_lines)
        wav_paths = sorted(wav_set)
        vid_paths = sorted(vid_set)
    else:
        wav_paths, vid_paths = list_split_media(split_name)

    if start_index > 0:
        wav_paths = wav_paths[start_index:]
        vid_paths = vid_paths[start_index:]
    if max_utterances > 0:
        wav_paths = wav_paths[:max_utterances]
        vid_paths = vid_paths[:max_utterances]

    stats = {
        "wav_total": len(wav_paths),
        "vid_total": len(vid_paths),
        "video_encodes": 0,
        "audio_clean_encodes": 0,
        "audio_noisy_encodes": 0,
    }

    print(
        f"[utterances] prefix={prefix} split={split_name} "
        f"wav={stats['wav_total']} video={stats['vid_total']} source={media_source}"
    )

    if not skip_video and vid_paths:
        stats["video_encodes"] = extract_videos(
            vid_paths,
            prefix,
            video_enc,
            read_frames_fn=read_frames_fn,
            max_frames=max_frames,
            prefetcher=prefetcher,
        )

    if not skip_audio and wav_paths:
        stats["audio_clean_encodes"] = extract_clean_audio_batch(
            wav_paths,
            prefix,
            qwen_enc,
            encode_jobs_fn,
            wav_chunk=audio_chunk_size,
            audio_prefetcher=audio_prefetcher,
            save_workers=audio_save_workers,
            audio_encoder=audio_encoder,
        )
        stats["audio_noisy_encodes"] = extract_noisy_audio_batch(
            wav_paths,
            prefix,
            qwen_enc,
            encode_jobs_fn,
            seed=seed,
            save_noisy_wav=save_noisy_wav,
            wav_chunk=max(32, audio_chunk_size // 2),
            audio_prefetcher=audio_prefetcher,
            save_workers=audio_save_workers,
            audio_encoder=audio_encoder,
        )

    print(
        f"[utterances] done video={stats['video_encodes']} "
        f"audio_clean={stats['audio_clean_encodes']} audio_noisy={stats['audio_noisy_encodes']}"
    )
    return stats
