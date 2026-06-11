"""
Phase 2: assemble pair-level npy dicts from precomputed utterance embeddings.

No GPU — only G2P text features and path wiring. Dataloaders load embeddings
via anc/com_audi_fea_path and anc/com_vide_fea_path at train/test time.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Iterable, List

import numpy as np
from tqdm import tqdm

from embed_paths import SNR_LIST, resolve_audio_embed_path, resolve_video_embed_path
from paths_config import data_list_dir, npy_dir


def build_data_dict(
    sample: dict,
    anc_phn,
    com_phn,
    anc_text_fea,
    com_text_fea,
    anc_vide_fea_path: str,
    com_vide_fea_path: str,
    anc_audi_fea_path: str,
    com_audi_fea_path: str,
    anc_text: str,
    com_text: str,
    anc_lip_path: str,
    com_lip_path: str,
    anc_wav_path: str,
    com_wav_path: str,
    video_feat_dim: int,
) -> dict:
    return {
        "anc_phn_list": anc_phn,
        "com_phn_list": com_phn,
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
        "video_feat_dim": video_feat_dim,
    }


def _require_embed(path: str, kind: str, source: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {kind} embedding for {source}\n  expected: {path}\n"
            "Run utterance extraction (phase 1) first."
        )


def pair_paths_for_sample(prefix: str, sample: dict, *, audio_encoder: str = "qwen3") -> dict:
    anc_wav = sample["anc_wav_path"]
    com_wav = sample["com_wav_path"]
    return {
        "anc_audi": resolve_audio_embed_path(prefix, anc_wav, snr=None, encoder=audio_encoder),
        "com_audi": resolve_audio_embed_path(prefix, com_wav, snr=None, encoder=audio_encoder),
        "anc_vide": resolve_video_embed_path(prefix, sample["anc_lip_path"]),
        "com_vide": resolve_video_embed_path(prefix, sample["com_lip_path"]),
    }


def sample_is_complete(
    npy_save_dir: str, anc_base: str, com_base: str, *, write_snr_pairs: bool = True
) -> bool:
    clean = os.path.join(npy_save_dir, f"{anc_base}+{com_base}.npy")
    if not os.path.exists(clean):
        return False
    if not write_snr_pairs:
        return True
    return all(
        os.path.exists(os.path.join(npy_save_dir, f"{anc_base}+{com_base}_{snr}db.npy"))
        for snr in SNR_LIST
    )


def write_shuf_scp(lines: List[str], scp_name: str, npy_save_dir: str) -> None:
    os.makedirs(data_list_dir(), exist_ok=True)
    for dest in (npy_save_dir, data_list_dir() + os.sep):
        p = os.path.join(dest, f"shuf_{scp_name}.scp")
        with open(p, "w") as f:
            f.writelines(lines)
        print(f"Wrote {p} ({len(lines)} lines)")


def rebuild_shuf_scp(scp_name: str, npy_save_dir: str) -> int:
    pat = re.compile(r"_-?\d+db\.npy$")
    clean = sorted(
        p for p in glob.glob(os.path.join(npy_save_dir, "*.npy")) if not pat.search(p)
    )
    lines = [p + "\n" for p in clean]
    write_shuf_scp(lines, scp_name, npy_save_dir)
    return len(lines)


def run_pair_build(
    *,
    prefix: str,
    scp_lines: Iterable[str],
    scp_out_name: str,
    g2p,
    video_feat_dim: int,
    start_index: int = 0,
    max_samples: int = 0,
    rebuild_scp: bool = True,
    require_embeddings: bool = True,
    audio_encoder: str = "qwen3",
    write_snr_pairs: bool = True,
) -> dict:
    lines = [ln.strip() for ln in scp_lines if ln.strip()]
    if start_index > 0:
        lines = lines[start_index:]
    if max_samples > 0:
        lines = lines[:max_samples]

    npy_save_dir = npy_dir(prefix) + os.sep
    os.makedirs(npy_save_dir, exist_ok=True)

    skipped = 0
    written_clean = 0
    written_snr = 0

    for line in tqdm(lines, desc=f"pairs/{prefix}"):
        sample = np.load(line, allow_pickle=True).item()
        anc_wav_p = sample["anc_wav_path"]
        com_wav_p = sample["com_wav_path"]
        anc_base = os.path.basename(anc_wav_p).replace(".wav", "")
        com_base = os.path.basename(com_wav_p).replace(".wav", "")

        if sample_is_complete(
            npy_save_dir, anc_base, com_base, write_snr_pairs=write_snr_pairs
        ):
            skipped += 1
            continue

        anc_text, com_text = sample["anc_text"], sample["com_text"]
        anc_phn, anc_txt_fea = g2p(anc_text), g2p.embedding(anc_text)
        com_phn, com_txt_fea = g2p(com_text), g2p.embedding(com_text)

        paths = pair_paths_for_sample(prefix, sample, audio_encoder=audio_encoder)
        if require_embeddings:
            _require_embed(paths["anc_audi"], "audio", anc_wav_p)
            _require_embed(paths["com_audi"], "audio", com_wav_p)
            _require_embed(paths["anc_vide"], "video", sample["anc_lip_path"])
            _require_embed(paths["com_vide"], "video", sample["com_lip_path"])

        anc_lip, com_lip = sample["anc_lip_path"], sample["com_lip_path"]

        clean_npy = os.path.join(npy_save_dir, f"{anc_base}+{com_base}.npy")
        if not os.path.exists(clean_npy):
            d = build_data_dict(
                sample,
                anc_phn,
                com_phn,
                anc_txt_fea,
                com_txt_fea,
                paths["anc_vide"],
                paths["com_vide"],
                paths["anc_audi"],
                paths["com_audi"],
                anc_text,
                com_text,
                anc_lip,
                com_lip,
                anc_wav_p,
                com_wav_p,
                video_feat_dim,
            )
            np.save(clean_npy, d)
            written_clean += 1

        for snr in SNR_LIST if write_snr_pairs else ():
            snr_npy = os.path.join(npy_save_dir, f"{anc_base}+{com_base}_{snr}db.npy")
            if os.path.exists(snr_npy):
                continue
            d = build_data_dict(
                sample,
                anc_phn,
                com_phn,
                anc_txt_fea,
                com_txt_fea,
                paths["anc_vide"],
                paths["com_vide"],
                paths["anc_audi"],
                paths["com_audi"],
                anc_text,
                com_text,
                anc_lip,
                com_lip,
                anc_wav_p,
                com_wav_p,
                video_feat_dim,
            )
            np.save(snr_npy, d)
            written_snr += 1

    n_shuf = 0
    if rebuild_scp:
        n_shuf = rebuild_shuf_scp(scp_out_name, npy_save_dir)

    stats = {
        "skipped": skipped,
        "written_clean": written_clean,
        "written_snr": written_snr,
        "shuf_lines": n_shuf,
    }
    print(
        f"[pairs] skipped={skipped} new_clean={written_clean} new_snr={written_snr} "
        f"shuf={n_shuf}"
    )
    return stats
