import os
import sys
import torch
import numpy as np
import random
import wave
from tqdm import tqdm
import torchvision
from g2p.g2p_en.g2p import G2p
from lipreading.video_encoder import GrayCropFlip, CNN_Resnet
from qwen_audio_encoder import QwenAudioEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import (
    LIPREADING_CKPT,
    NOISE_ROOT,
    PREFIX_CONFIG,
    data_list_dir,
    features_dir,
    noisy_wav_dir,
    npy_dir,
    raw_scp_path,
)

seed = 42
random.seed(seed)

prefix = "eval"
cfg = PREFIX_CONFIG[prefix]
scp_file = raw_scp_path(cfg["data_split"])
fea_save_dir = features_dir(prefix) + os.sep
npy_save_dir = npy_dir(prefix) + os.sep
noisy_wav_save_dir = noisy_wav_dir(prefix) + os.sep
scp_out_name = cfg["scp_name"]

snr_list = [5, 0, -5, -10]

noise_root = NOISE_ROOT
noise_list = ["Home", "Music", "TV", "Store", "WindAirCon", "WindFan", "babble_noise"]
noise_dir_map = {"Home": "GenHome", "Music": "GenMusic"}
choose_weights = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.70]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

qwen_enc = QwenAudioEncoder(model_id="Qwen/Qwen2-Audio-7B", device=device, max_frames=100)
g2p = G2p()

CNN_Resnet = CNN_Resnet(output_dim=256)
GrayCropFlip = GrayCropFlip(channel_input="rgb").to(device)
checkpoint_pretrain = torch.load(LIPREADING_CKPT, map_location=device)
CNN_Resnet.load_state_dict(checkpoint_pretrain)
CNN_Resnet.to(device)
CNN_Resnet.eval()
GrayCropFlip.eval()


def read_audio(wav_path):
    with wave.open(wav_path, "rb") as wf:
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

    dtype = np.int16 if sample_width == 2 else np.int32
    return sample_width, frame_rate, np.frombuffer(audio_data, dtype=dtype)


def write_audio(audio_data, audio_name):
    audio_data_int16 = (audio_data * 32768.0).clip(-32768, 32767).astype(np.int16)
    wave_file = wave.open(audio_name, "wb")
    wave_file.setparams((1, 2, 16000, len(audio_data_int16), "NONE", "NONE"))
    wave_file.writeframes(audio_data_int16.tobytes())
    wave_file.close()


def AudioEncoder(audio, qwen_enc=qwen_enc):
    audio_embed = qwen_enc.encode(audio)
    return audio_embed.numpy()


def TextEncoder(text, g2p=g2p):
    return g2p(text), torch.from_numpy(g2p.embedding(text)).numpy()


def VideoEncoder(lip, network=CNN_Resnet):
    with torch.no_grad():
        lip = lip.to(torch.float32)
        gray_lip, _ = GrayCropFlip(lip)
        batched_gray_lip = gray_lip.unsqueeze(1)
        return network(batched_gray_lip).detach().cpu().numpy()


def audioAddNoiseScale(clean_wav, noise_wav, snr):
    clean_wav = np.array(clean_wav, dtype=np.float32) / 32768.0
    noise_wav = np.array(noise_wav, dtype=np.float32) / 32768.0

    clean_power = np.mean(clean_wav ** 2)

    if len(noise_wav) > len(clean_wav):
        start_idx = random.randint(0, len(noise_wav) - len(clean_wav))
        noise_wav = noise_wav[start_idx : start_idx + len(clean_wav)]
    else:
        noise_wav = np.pad(noise_wav, (0, len(clean_wav) - len(noise_wav)), "wrap")

    noise_power = np.mean(noise_wav ** 2)
    if noise_power == 0:
        return clean_wav

    scaling_factor = np.sqrt(clean_power / (10 ** (snr / 10) * noise_power))
    return clean_wav + (noise_wav * scaling_factor)


def audi_fea_path_canonical(snr=None):
    """Paths use /wav/ so train/test dataloaders can swap to /wav_{snr}db/."""
    sub = "wav" if snr is None else f"wav_{snr}db"
    return os.path.join(fea_save_dir, sub)


def build_data_dict(
    sample,
    anc_phn_list,
    com_phn_list,
    anc_text_fea,
    com_text_fea,
    anc_vide_fea_path,
    com_vide_fea_path,
    anc_audi_fea_path,
    com_audi_fea_path,
    anc_text,
    com_text,
    anc_lip_path,
    com_lip_path,
    anc_wav_path,
    com_wav_path,
):
    return {
        "anc_phn_list": anc_phn_list,
        "com_phn_list": com_phn_list,
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
    }


def write_shuf_scp(lines, scp_name):
    os.makedirs(data_list_dir(), exist_ok=True)
    os.makedirs(os.path.dirname(npy_save_dir.rstrip(os.sep)), exist_ok=True)
    for dest in (npy_save_dir, data_list_dir() + os.sep):
        shuf_path = os.path.join(dest, f"shuf_{scp_name}.scp")
        with open(shuf_path, "w") as f:
            f.writelines(lines)
        print(f"Wrote {shuf_path} ({len(lines)} lines)")


with open(scp_file) as f:
    lines = f.readlines()

shuf_scp_lines = []

for line in tqdm(lines, desc="Processing samples"):
    line = line.strip()
    sample = np.load(line, allow_pickle=True).item()

    com_wav_path, anc_wav_path = sample["com_wav_path"], sample["anc_wav_path"]
    anc_lip_path, com_lip_path = sample["anc_lip_path"], sample["com_lip_path"]
    anc_text, com_text = sample["anc_text"], sample["com_text"]

    anc_phn_list, anc_text_fea = TextEncoder(anc_text)
    com_phn_list, com_text_fea = TextEncoder(com_text)

    vid_base_dir = os.path.join(fea_save_dir, "lip")
    anc_vide_fea_path = os.path.join(
        vid_base_dir, anc_lip_path.lstrip("/").replace(".mp4", ".npy")
    )
    com_vide_fea_path = os.path.join(
        vid_base_dir, com_lip_path.lstrip("/").replace(".mp4", ".npy")
    )

    if not os.path.exists(anc_vide_fea_path):
        vid_frames, _, _ = torchvision.io.read_video(anc_lip_path, pts_unit="sec")
        anc_vide_fea = VideoEncoder(vid_frames.cuda())
        os.makedirs(os.path.dirname(anc_vide_fea_path), exist_ok=True)
        np.save(anc_vide_fea_path, anc_vide_fea)

    if not os.path.exists(com_vide_fea_path):
        vid_frames, _, _ = torchvision.io.read_video(com_lip_path, pts_unit="sec")
        com_vide_fea = VideoEncoder(vid_frames.cuda())
        os.makedirs(os.path.dirname(com_vide_fea_path), exist_ok=True)
        np.save(com_vide_fea_path, com_vide_fea)

    _, _, clean_com_wav = read_audio(com_wav_path)
    _, _, clean_anc_wav = read_audio(anc_wav_path)

    clean_audi_dir = audi_fea_path_canonical(snr=None)
    com_clean_fea = os.path.join(
        clean_audi_dir, os.path.basename(com_wav_path).replace(".wav", ".npy")
    )
    anc_clean_fea = os.path.join(
        clean_audi_dir, os.path.basename(anc_wav_path).replace(".wav", ".npy")
    )
    if not os.path.exists(com_clean_fea):
        os.makedirs(os.path.dirname(com_clean_fea), exist_ok=True)
        np.save(com_clean_fea, AudioEncoder(clean_com_wav / 32768.0))
    if not os.path.exists(anc_clean_fea):
        os.makedirs(os.path.dirname(anc_clean_fea), exist_ok=True)
        np.save(anc_clean_fea, AudioEncoder(clean_anc_wav / 32768.0))

    anc_base = os.path.basename(anc_wav_path).replace(".wav", "")
    com_base = os.path.basename(com_wav_path).replace(".wav", "")
    clean_dict_name = f"{anc_base}+{com_base}.npy"
    clean_save_path = os.path.join(npy_save_dir, clean_dict_name)
    if not os.path.exists(clean_save_path):
        clean_dict = build_data_dict(
            sample,
            anc_phn_list,
            com_phn_list,
            anc_text_fea,
            com_text_fea,
            anc_vide_fea_path,
            com_vide_fea_path,
            anc_clean_fea,
            com_clean_fea,
            anc_text,
            com_text,
            anc_lip_path,
            com_lip_path,
            anc_wav_path,
            com_wav_path,
        )
        os.makedirs(os.path.dirname(clean_save_path), exist_ok=True)
        np.save(clean_save_path, clean_dict)
        shuf_scp_lines.append(clean_save_path + "\n")

    for snr in snr_list:
        seed += 1

        noise_name = random.choices(noise_list, weights=choose_weights, k=1)[0]
        noise_corpus = os.path.join(noise_root, noise_dir_map.get(noise_name, noise_name))
        noise_wav_list = [w for w in os.listdir(noise_corpus) if w.endswith(".wav")]
        if not noise_wav_list:
            raise FileNotFoundError(f"No .wav files in {noise_corpus}")
        _, _, noise_wav = read_audio(
            os.path.join(noise_corpus, random.choice(noise_wav_list))
        )

        noisy_com_wav = audioAddNoiseScale(clean_com_wav, noise_wav, snr)
        noisy_anc_wav = audioAddNoiseScale(clean_anc_wav, noise_wav, snr)

        wav_save_dir = os.path.join(noisy_wav_save_dir, f"{prefix}_{snr}db")
        audi_dir = audi_fea_path_canonical(snr=snr)

        com_noisy_path = os.path.join(wav_save_dir, os.path.basename(com_wav_path))
        anc_noisy_path = os.path.join(wav_save_dir, os.path.basename(anc_wav_path))

        com_audi_fea_path = os.path.join(
            audi_dir, os.path.basename(com_wav_path).replace(".wav", ".npy")
        )
        anc_audi_fea_path = os.path.join(
            audi_dir, os.path.basename(anc_wav_path).replace(".wav", ".npy")
        )

        if not os.path.exists(com_audi_fea_path):
            os.makedirs(os.path.dirname(com_audi_fea_path), exist_ok=True)
            np.save(com_audi_fea_path, AudioEncoder(noisy_com_wav))
            os.makedirs(os.path.dirname(com_noisy_path), exist_ok=True)
            write_audio(noisy_com_wav, com_noisy_path)

        if not os.path.exists(anc_audi_fea_path):
            os.makedirs(os.path.dirname(anc_audi_fea_path), exist_ok=True)
            np.save(anc_audi_fea_path, AudioEncoder(noisy_anc_wav))
            os.makedirs(os.path.dirname(anc_noisy_path), exist_ok=True)
            write_audio(noisy_anc_wav, anc_noisy_path)

        # Canonical /wav/ paths for dataloader SNR swapping at train/test time
        com_audi_canonical = com_clean_fea
        anc_audi_canonical = anc_clean_fea

        dict_name = f"{anc_base}+{com_base}_{snr}db.npy"
        save_path = os.path.join(npy_save_dir, dict_name)
        data_dict = build_data_dict(
            sample,
            anc_phn_list,
            com_phn_list,
            anc_text_fea,
            com_text_fea,
            anc_vide_fea_path,
            com_vide_fea_path,
            anc_audi_canonical,
            com_audi_canonical,
            anc_text,
            com_text,
            anc_lip_path,
            com_lip_path,
            anc_wav_path,
            com_wav_path,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, data_dict)

write_shuf_scp(shuf_scp_lines, scp_out_name)
print("Done.")
