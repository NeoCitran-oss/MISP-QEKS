"""Build a capped shuf_train.scp from finished SigLIP2 train feature pairs."""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import data_list_dir, npy_dir, VIDEO_LIP_SUBDIR_SIGLIP2


def pair_has_snr_audio(com_audi_fea_path: str, snrs: list[int]) -> bool:
    for snr in snrs:
        snr_path = com_audi_fea_path.replace("/wav/", f"/wav_{snr}db/")
        if not os.path.isfile(snr_path):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Write shuf_<name>.scp with up to N complete SigLIP2 train pairs."
    )
    parser.add_argument("--prefix", type=str, default="train")
    parser.add_argument("--max_pairs", type=int, default=50000)
    parser.add_argument("--scp_name", type=str, default="train")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for shuf_<scp_name>.scp (default: data_list/)",
    )
    parser.add_argument(
        "--train_snrs",
        type=str,
        default="5,0,-5,-10",
        help="Require precomputed com audio features at these SNRs",
    )
    parser.add_argument(
        "--allow_cnn_video",
        action="store_true",
        help="Do not require lip_siglip2 in anc_vide_fea_path",
    )
    args = parser.parse_args()

    snrs = [int(x) for x in args.train_snrs.split(",")]
    npy_root = npy_dir(args.prefix)
    pattern = re.compile(r"_-?\d+db\.npy$")
    clean_npys = sorted(
        p for p in glob.glob(os.path.join(npy_root, "*.npy")) if not pattern.search(p)
    )

    selected = []
    skipped_video = 0
    skipped_snr = 0

    for path in clean_npys:
        if len(selected) >= args.max_pairs:
            break

        data = np.load(path, allow_pickle=True).item()
        video_path = data.get("anc_vide_fea_path", "")
        if not args.allow_cnn_video and VIDEO_LIP_SUBDIR_SIGLIP2 not in video_path:
            skipped_video += 1
            continue
        if not pair_has_snr_audio(data["com_audi_fea_path"], snrs):
            skipped_snr += 1
            continue
        selected.append(path)

    out_dir = args.output_dir or data_list_dir()
    os.makedirs(out_dir, exist_ok=True)
    scp_path = os.path.join(out_dir, f"shuf_{args.scp_name}.scp")
    with open(scp_path, "w", encoding="utf-8") as f:
        f.writelines(p + "\n" for p in selected)

    print(f"Scan:   {len(clean_npys)} clean npy under {npy_root}")
    print(f"Wrote:  {len(selected)} pairs -> {scp_path}")
    print(f"Skip:   {skipped_video} (not SigLIP2 video), {skipped_snr} (missing SNR audio)")
    if len(selected) < args.max_pairs:
        print(
            f"NOTE: Only {len(selected)} ready pairs (wanted {args.max_pairs}). "
            "Let extraction run longer or lower --max_pairs.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
