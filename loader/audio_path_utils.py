"""Resolve precomputed audio feature paths for MISP-QEKS dataloaders."""
import re


def resolve_com_audi_fea_path(path: str, snr):
    """Map stored com_audi_fea_path to the file for the requested SNR (or clean)."""
    if snr is None:
        return path
    snr_tag = f"wav_{snr}db"
    if f"/{snr_tag}/" in path or path.startswith(f"{snr_tag}/"):
        return path
    if "/wav/" in path:
        return path.replace("/wav/", f"/{snr_tag}/")
    return path


def filter_scp_for_clean(files_scp):
    """Keep clean npy entries (no _<snr>db.npy suffix)."""
    return [p for p in files_scp if not re.search(r"_-?\d+db\.npy$", p)]
