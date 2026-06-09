"""Rebuild data_list/shuf_<scp>.scp from clean npy dicts on disk."""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import PREFIX_CONFIG, data_list_dir, npy_dir


def rebuild(prefix: str) -> int:
    cfg = PREFIX_CONFIG[prefix]
    npy_save_dir = npy_dir(prefix)
    scp_name = cfg["scp_name"]
    pattern = re.compile(r"_-?\d+db\.npy$")
    clean_npys = sorted(
        p for p in glob.glob(os.path.join(npy_save_dir, "*.npy")) if not pattern.search(p)
    )
    lines = [p + "\n" for p in clean_npys]

    os.makedirs(data_list_dir(), exist_ok=True)
    for dest in (npy_save_dir, data_list_dir() + os.sep):
        shuf_path = os.path.join(dest, f"shuf_{scp_name}.scp")
        with open(shuf_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Wrote {shuf_path} ({len(lines)} lines)")

    return len(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="train", choices=list(PREFIX_CONFIG.keys()))
    args = parser.parse_args()
    n = rebuild(args.prefix)
    print(f"Done. {n} clean pairs for prefix={args.prefix}")
