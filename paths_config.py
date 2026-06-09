"""Shared paths for MISP-QEKS on linna@tars (override via env vars)."""
import os
import sys

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
# XEQ-Matcher Vide_Proj default input size (CNN lip and projected SigLIP2).
MATCHER_VIDEO_FEAT_DIM = 256
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


def results_dir() -> str:
    """Pipeline logs (feature extraction, train, test). Created on first use."""
    path = os.path.join(MISP_BASELINE, "results")
    os.makedirs(path, exist_ok=True)
    return path


def siglip2_log_path(prefix: str) -> str:
    return os.path.join(results_dir(), f"siglip2_{prefix}.log")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_run_log(log_path: str) -> str:
    """Mirror stdout/stderr to log_path (append). Returns the resolved path."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path


def hf_cache_root() -> str:
    """Hugging Face / transformers download cache (keep off laptop home dir)."""
    return os.environ.get("HF_CACHE_ROOT", os.path.join(MISP_BASELINE, "hf_cache"))


def configure_scratch_storage() -> str:
    """
    Redirect model caches away from ~/.cache and ~/nltk_data to scratch.

    Call this before importing transformers or loading Qwen / SigLIP 2.
    """
    root = hf_cache_root()
    hub = os.path.join(root, "hub")
    transformers_cache = os.path.join(root, "transformers")
    torch_home = os.path.join(root, "torch")
    nltk_data = os.path.join(root, "nltk_data")

    for path in (root, hub, transformers_cache, torch_home, nltk_data):
        os.makedirs(path, exist_ok=True)

    os.environ["HF_HOME"] = root
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["TORCH_HOME"] = torch_home
    os.environ["NLTK_DATA"] = nltk_data
    return root


def is_tars_scratch_layout() -> bool:
    return MISP_BASELINE.startswith("/local/scratch") and MISP_DATA.startswith(
        "/local/scratch"
    )


def ensure_scratch_execution(allow_local: bool = False) -> None:
    """Refuse to run on a laptop/home layout unless explicitly overridden."""
    if allow_local:
        return
    if not is_tars_scratch_layout():
        print(
            "ERROR: This job must run on tars scratch, not your local computer.\n"
            f"  MISP_BASELINE = {MISP_BASELINE}\n"
            f"  MISP_DATA     = {MISP_DATA}\n"
            "SSH to linna@tars.cl.uzh.ch and run there, or pass --allow-local "
            "(not recommended — fills your home disk with HF model caches).",
            file=sys.stderr,
        )
        sys.exit(1)
