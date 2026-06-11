"""Resolve precomputed audio feature paths for MISP-QEKS dataloaders."""
import re


def resolve_com_audi_fea_path(path: str, snr):
    """Map stored com_audi_fea_path to the file for the requested SNR (or clean)."""
    if snr is None:
        return path
    # Qwen3 mirrored layout: features/<prefix>/wav_qwen3/data/<split>/wav/x.npy
    #   noisy variant:       features/<prefix>/wav_qwen3_<snr>db/data/<split>/wav/x.npy
    if re.search(r"/wav_qwen3_-?\d+db/", path):
        return path
    if "/wav_qwen3/" in path:
        return path.replace("/wav_qwen3/", f"/wav_qwen3_{snr}db/")
    # Legacy Qwen2 flat layout: features/<prefix>/wav/x.npy -> wav_<snr>db/x.npy
    snr_tag = f"wav_{snr}db"
    if f"/{snr_tag}/" in path or path.startswith(f"{snr_tag}/"):
        return path
    if "/wav/" in path:
        return path.replace("/wav/", f"/{snr_tag}/")
    return path


def filter_scp_for_clean(files_scp):
    """Keep clean npy entries (no _<snr>db.npy suffix)."""
    return [p for p in files_scp if not re.search(r"_-?\d+db\.npy$", p)]
