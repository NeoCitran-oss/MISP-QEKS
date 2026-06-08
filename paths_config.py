"""Shared paths for MISP-QEKS on linna@tars (override via env vars)."""
import os

MISP_DATA = os.environ.get(
    "MISP_DATA_ROOT",
    "/local/scratch/linna/MISP/MISP_data/MISP-QEKS",
)
MISP_BASELINE = os.environ.get(
    "MISP_BASELINE_ROOT",
    "/local/scratch/linna/MISP/MISP_baseline/MISP-QEKS",
)

NOISE_ROOT = os.path.join(MISP_DATA, "data", "noise")
PRETRAIN_MODEL_ROOT = os.path.join(MISP_DATA, "model")
LIPREADING_CKPT = os.path.join(
    PRETRAIN_MODEL_ROOT, "lipreading", "lipreading_LRW_0.8018.pt"
)

# Google SigLIP 2 video encoder (see data_prepare/siglip2_video_encoder.py)
SIGLIP2_MODEL_ID = os.environ.get(
    "SIGLIP2_MODEL_ID", "google/siglip2-base-patch16-224"
)
VIDEO_LIP_SUBDIR_CNN = "lip"
VIDEO_LIP_SUBDIR_SIGLIP2 = "lip_siglip2"
BASELINE_CKPT_DIR = os.path.join(MISP_DATA, "train", "model")

# feature-extraction prefix -> preprocess split / dataloader scp stem
PREFIX_CONFIG = {
    "train": {
        "data_split": "train",
        "scp_name": "train",
    },
    "dev": {
        "data_split": "dev_seen",
        "scp_name": "dev",
    },
    "eval": {
        "data_split": "eval_seen",
        "scp_name": "eval_inset",
    },
    "eval_unseen": {
        "data_split": "eval_unseen",
        "scp_name": "eval_outset",
    },
}


def data_split_dir(split_name: str) -> str:
    return os.path.join(MISP_DATA, "data", split_name)


def raw_dict_dir(split_name: str) -> str:
    return os.path.join(MISP_DATA, "raw_dicts", split_name)


def raw_scp_path(split_name: str) -> str:
    return os.path.join(raw_dict_dir(split_name), f"raw_{split_name}.scp")


def features_dir(prefix: str) -> str:
    return os.path.join(MISP_BASELINE, "features", prefix)


def npy_dir(prefix: str) -> str:
    return os.path.join(MISP_BASELINE, "npy", prefix)


def noisy_wav_dir(prefix: str) -> str:
    return os.path.join(MISP_BASELINE, "noisy_wav", prefix)


def data_list_dir() -> str:
    return os.path.join(MISP_BASELINE, "data_list")
