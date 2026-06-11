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
    features_dir,
    noisy_wav_dir,
    npy_dir,
    raw_scp_path,
)

seed = 42
random.seed(seed)

prefix = "eval"
_cfg = PREFIX_CONFIG[prefix]
scp = raw_scp_path(_cfg["data_split"])
fea_save_dir = features_dir(prefix) + "/"
npy_save_dir = npy_dir(prefix) + "/"
noisy_wav_save_dir = noisy_wav_dir(prefix) + "/"

snr_list = [5, 0, -5, -10]

noise_root = NOISE_ROOT
noise_list = ['Home',
              'Music',
              'TV',
              'Store',
              'WindAirCon',
              'WindFan',
              'babble_noise']
noise_dir_map = {
    'Home': 'GenHome',
    'Music': 'GenMusic',
}

choose_weights = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.70]

# wav encoder (Qwen2-Audio audio tower, no LLM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
qwen_enc = QwenAudioEncoder(model_id="Qwen/Qwen2-Audio-7B", device=device, max_frames=100)

# text encoder
g2p = G2p()

# video encoder
CNN_Resnet = CNN_Resnet(output_dim=256)
GrayCropFlip = GrayCropFlip(channel_input='rgb')
GrayCropFlip.to(device)
checkpoint_pretrain = torch.load(LIPREADING_CKPT, map_location=device)
CNN_Resnet.load_state_dict(checkpoint_pretrain)
CNN_Resnet.to(device)


def read_audio(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate() 
        n_frames = wf.getnframes() 
        audio_data = wf.readframes(n_frames)
    
    if sample_width == 2:  # if 16-bit PCM
        dtype = np.int16
    elif sample_width == 4:  # if 32-bit PCM
        dtype = np.int32
    else:
        raise ValueError("Unsupported sample_width: {}".format(sample_width))
    
    audio_array = np.frombuffer(audio_data, dtype=dtype)

    return sample_width, frame_rate, audio_array

def write_audio(audio_data, audio_name):
    nchannels = 1
    sampwidth = 2
    framerate = 16000
    nframes = len(audio_data)
    wave_file = wave.open(audio_name, "wb")
    wave_file.setparams(((nchannels, sampwidth, framerate, nframes, 'NONE', 'NONE')))
    wave_file.writeframes(np.array(audio_data, dtype="int16").tobytes())
    wave_file.close()
    return 0

def AudioEncoder(audio, qwen_enc=qwen_enc):
    # `audio` carries int16 magnitude as float32; normalize to [-1, 1] for the
    # Qwen feature extractor.
    audio = np.asarray(audio, dtype=np.float32) / 32768.0
    audio_embed = qwen_enc.encode(audio)  # [1, <=100, 1280] [t, c]

    return audio_embed.numpy()

def TextEncoder(text, g2p=g2p):
    ang2p = g2p(text)
    an_embed = g2p.embedding(text)
    an_embed = torch.from_numpy(an_embed)

    return ang2p, an_embed

def VideoEncoder(lip, network=CNN_Resnet):
    lip = lip.to(torch.float32)
    gray_lip, _ = GrayCropFlip(lip)
    batched_gray_lip = gray_lip.unsqueeze(1)
    lip_embed = network(batched_gray_lip)
    return lip_embed

def audioAddNoiseScale(clean_wav, noise_wav, snr):
    noise_power = 0

    clean_wav = np.array(clean_wav, dtype=np.float32)
    noise_wav = np.array(noise_wav, dtype=np.float32)

    clean_power = np.mean(clean_wav ** 2)

    if len(noise_wav) > len(clean_wav):
        start_idx  = random.randint(0, len(noise_wav) - len(clean_wav))
        noise_wav = noise_wav[start_idx : start_idx + len(clean_wav)]
    else:
        noise_wav = np.pad(noise_wav, (0, len(clean_wav) - len(noise_wav)), 'wrap')

    noise_power = np.mean(noise_wav ** 2)

    if noise_power == 0:
        print("Getting noise_power=0, skip adding noise")
        return clean_wav
    
    # write_audio(clean_wav, './clean.wav')
    # write_audio(noise_wav, './noise.wav')        

    scaling_factor = np.sqrt(clean_power / (10**(snr / 10) * noise_power))
    noise_wav = noise_wav * scaling_factor
    noisy_wav = clean_wav + noise_wav

    return noisy_wav

with open(scp) as f:
    lines = f.readlines()

for snr in snr_list:
    wav_save_dir = f'{noisy_wav_save_dir}{prefix}_{snr}db/'
    
    seed += 1

    for line in tqdm(lines):


        line = line.strip()
        sample = np.load(line.strip(), allow_pickle=True).item()
        com_wav_path = sample['com_wav_path']
        anc_wav_path = sample['anc_wav_path']
        
        sample_width, frame_rate, clean_com_wav = read_audio(com_wav_path)
        sample_width, frame_rate, clean_anc_wav = read_audio(anc_wav_path)
        
        noise_name = random.choices(noise_list, weights=choose_weights, k=1)[0]
        noise_corpus = os.path.join(noise_root, noise_dir_map.get(noise_name, noise_name))

        noise_wav_list = [wav for wav in os.listdir(noise_corpus) if wav.endswith('.wav')]
        if not noise_wav_list:
            raise FileNotFoundError(f"No .wav files found in noise directory: {noise_corpus}")
        noise_wav_path = os.path.join(noise_corpus, random.choice(noise_wav_list))
        _, _, noise_wav = read_audio(noise_wav_path)

        noisy_com_wav = audioAddNoiseScale(clean_com_wav, noise_wav, snr)
        noisy_anc_wav = audioAddNoiseScale(clean_anc_wav, noise_wav, snr)

        com_noisy_path = wav_save_dir + '/'.join(com_wav_path.split('/')[-2:]).split('.')[0] + '.wav'
        anc_noisy_path = wav_save_dir + '/'.join(anc_wav_path.split('/')[-2:]).split('.')[0] + '.wav'

        audi_fea_path = fea_save_dir + f'wav_{snr}db/'
        com_audi_fea_path = audi_fea_path + com_wav_path.replace('.wav', '.npy')
        anc_audi_fea_path = audi_fea_path + anc_wav_path.replace('.wav', '.npy')


        anc_text = sample['anc_text']
        com_text = sample['com_text']

        anc_lip_path  = sample['anc_lip_path']
        com_lip_path  = sample['com_lip_path']

        label = sample['label']
        datatype = sample['type']
        new_file_scp = []


        anc_vide_fea_path = anc_audi_fea_path.replace('/wav/', '/lip/')
        com_vide_fea_path = com_audi_fea_path.replace('/wav/', '/lip/')

        # audio feature
        if not os.path.exists(com_audi_fea_path):
            com_audi_fea = AudioEncoder(noisy_com_wav)
            directory = os.path.dirname(com_audi_fea_path)
            os.makedirs(directory, exist_ok=True)
            np.save(com_audi_fea_path, com_audi_fea)

            wav_directory = os.path.dirname(com_noisy_path)
            os.makedirs(wav_directory, exist_ok=True)
            write_audio(noisy_com_wav, com_noisy_path)

        if not os.path.exists(anc_audi_fea_path):
            anc_audi_fea = AudioEncoder(noisy_anc_wav)
            directory = os.path.dirname(anc_audi_fea_path)
            os.makedirs(directory, exist_ok=True)
            np.save(anc_audi_fea_path, anc_audi_fea)

            wav_directory = os.path.dirname(anc_noisy_path)
            os.makedirs(wav_directory, exist_ok=True)
            write_audio(noisy_anc_wav, anc_noisy_path)

        # video feature
        if not os.path.exists(anc_vide_fea_path):
            vid_frames, _, _ = torchvision.io.read_video(anc_lip_path, pts_unit='sec')
            anc_lip = vid_frames.cuda()
            anc_vide_fea = VideoEncoder(anc_lip).detach().cpu().numpy()
            directory = os.path.dirname(anc_vide_fea_path)
            os.makedirs(directory, exist_ok=True)
            np.save(anc_vide_fea_path, anc_vide_fea)

        if not os.path.exists(com_vide_fea_path):
            vid_frames, _, _ = torchvision.io.read_video(com_lip_path, pts_unit='sec')
            com_lip = vid_frames.cuda()
            com_vide_fea = VideoEncoder(com_lip).detach().cpu().numpy()
            directory = os.path.dirname(com_vide_fea_path)
            os.makedirs(directory, exist_ok=True)
            np.save(com_vide_fea_path, com_vide_fea)

        # Text feature
        anc_phn_list, anc_text_fea = TextEncoder(anc_text)
        com_phn_list, com_text_fea = TextEncoder(com_text)

        data_dict={
                'anc_phn_list':anc_phn_list,
                'com_phn_list':com_phn_list,
                'anc_text_fea':anc_text_fea,
                'com_text_fea':com_text_fea,
                'anc_vide_fea_path':anc_vide_fea_path, 
                'com_vide_fea_path':com_vide_fea_path, 
                'anc_audi_fea_path':anc_audi_fea_path, 
                'com_audi_fea_path':com_audi_fea_path, 
                'type':datatype, 
                'label':label, 
                'anc_text':anc_text, 
                'com_text':com_text, 
                'anc_lip_path':anc_lip_path, 
                'com_lip_path':com_lip_path,
                'anc_wav_path':anc_wav_path, 
                'com_wav_path':com_wav_path
                }
            
        name = anc_wav_path + '+' + com_wav_path
        save_path = npy_save_dir + name + '.npy'
        directory = os.path.dirname(save_path)
        os.makedirs(directory, exist_ok=True)
        np.save(save_path, data_dict)


# env: py39 + torch1131 + cuda117
