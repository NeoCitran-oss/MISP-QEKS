"""Build raw_dicts + .scp for any MISP-QEKS split."""
import argparse
import glob
import os

import numpy as np

from paths_config import data_split_dir, raw_dict_dir


def build_split(split: str) -> int:
    base_dir = data_split_dir(split)
    output_dir = raw_dict_dir(split)
    os.makedirs(output_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(base_dir, "wav", "*_query.wav"))
    scp_lines = []
    print(f"[{split}] processing {len(wav_files)} query wav files...")

    for query_wav in wav_files:
        filename = os.path.basename(query_wav)
        pair_id = filename.replace("_query.wav", "")

        enroll_wav = os.path.join(base_dir, "wav", f"{pair_id}_enroll.wav")
        query_vid = os.path.join(base_dir, "mp4", f"{pair_id}_query.mp4")
        enroll_vid = os.path.join(base_dir, "mp4", f"{pair_id}_enroll.mp4")
        label_txt = os.path.join(base_dir, "label", f"{pair_id}.txt")

        if not os.path.exists(query_vid):
            query_vid = query_vid.replace(".mp4", ".m4p")
            enroll_vid = enroll_vid.replace(".mp4", ".m4p")

        if not (
            os.path.exists(enroll_wav)
            and os.path.exists(query_vid)
            and os.path.exists(label_txt)
        ):
            continue

        with open(label_txt, "r") as f:
            content = f.read().strip()

        anc_text, com_text, label_val = "", "", 1
        for part in content.split():
            if part.startswith("enrollment:"):
                anc_text = part.split(":")[1]
            elif part.startswith("query:"):
                com_text = part.split(":")[1]
            elif part.startswith("label:"):
                label_val = int(part.split(":")[1])

        data_dict = {
            "anc_wav_path": enroll_wav,
            "anc_lip_path": enroll_vid,
            "anc_text": anc_text,
            "anc_phn_list": [anc_text],
            "com_wav_path": query_wav,
            "com_lip_path": query_vid,
            "com_text": com_text,
            "com_phn_list": [com_text],
            "label": label_val,
            "type": split,
        }

        save_path = os.path.join(output_dir, f"{pair_id}.npy")
        np.save(save_path, data_dict)
        scp_lines.append(save_path + "\n")

    scp_path = os.path.join(output_dir, f"raw_{split}.scp")
    with open(scp_path, "w") as f:
        f.writelines(scp_lines)

    print(f"[{split}] wrote {len(scp_lines)} dicts -> {scp_path}")
    return len(scp_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        type=str,
        default="eval_seen,eval_unseen",
        help="Comma-separated split names under data/ (e.g. train,eval_seen)",
    )
    args = parser.parse_args()

    total = 0
    for split in args.splits.split(","):
        total += build_split(split.strip())
    print(f"Done. Total pairs: {total}")
