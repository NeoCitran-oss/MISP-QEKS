"""Map source media paths to mirrored precomputed embedding .npy paths."""
from __future__ import annotations

import glob
import os
from typing import Iterable, List, Optional, Set, Tuple

from paths_config import MISP_DATA, data_split_dir, features_dir

VIDEO_LIP_SUBDIR = "lip_siglip2"
AUDIO_CLEAN_SUBDIR = "wav"
SNR_LIST = (3, 6, 9)  # train noisy SNRs (matches loader default)
AUDIO_ENCODER_SUBDIRS = {
    "qwen2": "wav",
    "qwen3": "wav_qwen3",
}


def audio_modality_subdir(encoder: str = "qwen2", snr: Optional[int] = None) -> str:
    clean = AUDIO_ENCODER_SUBDIRS.get(encoder, AUDIO_CLEAN_SUBDIR)
    if snr is None:
        return clean
    if encoder == "qwen3":
        return f"wav_qwen3_{snr}db"
    return f"wav_{snr}db"


def _strip_leading_sep(path: str) -> str:
    return path.lstrip("/\\")


def source_rel_path(source_path: str) -> str:
    """Path of a media file relative to MISP_DATA (e.g. data/train/wav/foo.wav)."""
    source_path = os.path.normpath(source_path)
    data_root = os.path.normpath(MISP_DATA)
    if source_path.startswith(data_root + os.sep):
        return source_path[len(data_root) + 1 :]
    return _strip_leading_sep(source_path)


def embed_path(
    prefix: str,
    modality_subdir: str,
    source_path: str,
    *,
    features_root: Optional[str] = None,
) -> str:
    """
    Mirror source layout under features/<prefix>/<modality_subdir>/.

    Example:
      source  .../MISP_data/.../data/train/wav/pair_001_enroll.wav
      embed   .../features/train/wav/data/train/wav/pair_001_enroll.npy
    """
    root = features_root or features_dir(prefix)
    rel = source_rel_path(source_path)
    stem, _ = os.path.splitext(rel)
    return os.path.join(root, modality_subdir, stem + ".npy")


def audio_embed_path(
    prefix: str,
    wav_path: str,
    snr: Optional[int] = None,
    *,
    encoder: str = "qwen2",
) -> str:
    sub = audio_modality_subdir(encoder, snr)
    return embed_path(prefix, sub, wav_path)


def video_embed_path(prefix: str, video_path: str) -> str:
    return embed_path(prefix, VIDEO_LIP_SUBDIR, video_path)


def legacy_audio_embed_path(
    prefix: str,
    wav_path: str,
    snr: Optional[int] = None,
    *,
    encoder: str = "qwen2",
) -> str:
    """Old layout: features/<prefix>/wav/<basename>.npy (kept for resume)."""
    sub = audio_modality_subdir(encoder, snr)
    base = os.path.basename(wav_path).replace(".wav", ".npy")
    return os.path.join(features_dir(prefix), sub, base)


def resolve_audio_embed_path(
    prefix: str,
    wav_path: str,
    snr: Optional[int] = None,
    *,
    encoder: str = "qwen2",
) -> str:
    """Prefer mirrored path; fall back to legacy basename layout if present."""
    mirrored = audio_embed_path(prefix, wav_path, snr, encoder=encoder)
    if os.path.exists(mirrored):
        return mirrored
    legacy = legacy_audio_embed_path(prefix, wav_path, snr, encoder=encoder)
    if os.path.exists(legacy):
        return legacy
    return mirrored


def audio_embedding_exists(
    prefix: str,
    wav_path: str,
    snr: Optional[int] = None,
    *,
    encoder: str = "qwen2",
) -> bool:
    """True if either mirrored or legacy audio embedding already exists."""
    return os.path.exists(audio_embed_path(prefix, wav_path, snr, encoder=encoder)) or os.path.exists(
        legacy_audio_embed_path(prefix, wav_path, snr, encoder=encoder)
    )


def resolve_video_embed_path(prefix: str, video_path: str) -> str:
    mirrored = video_embed_path(prefix, video_path)
    if os.path.exists(mirrored):
        return mirrored
    legacy = os.path.join(
        features_dir(prefix),
        VIDEO_LIP_SUBDIR,
        _strip_leading_sep(video_path).replace(".mp4", ".npy").replace(".m4p", ".npy"),
    )
    if os.path.exists(legacy):
        return legacy
    return mirrored


def list_split_media(split_name: str) -> Tuple[List[str], List[str]]:
    """All wav and video files under data/<split>/."""
    base = data_split_dir(split_name)
    wavs = sorted(glob.glob(os.path.join(base, "wav", "*.wav")))
    vids = sorted(glob.glob(os.path.join(base, "mp4", "*.mp4")))
    vids += sorted(glob.glob(os.path.join(base, "mp4", "*.m4p")))
    return wavs, vids


def collect_utterances_from_scp(scp_lines: Iterable[str]) -> Tuple[Set[str], Set[str]]:
    """Unique wav / video paths referenced by raw pair dicts."""
    import numpy as np

    wavs: Set[str] = set()
    vids: Set[str] = set()
    for line in scp_lines:
        line = line.strip()
        if not line:
            continue
        sample = np.load(line, allow_pickle=True).item()
        wavs.add(sample["anc_wav_path"])
        wavs.add(sample["com_wav_path"])
        vids.add(sample["anc_lip_path"])
        vids.add(sample["com_lip_path"])
    return wavs, vids
