from pathlib import Path
import sys, os
import pandas as pd
from torch.utils.data import Dataset
import torch

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import librosa
import difflib
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
# Reuse the shared Qwen2-Audio audio-encoder backbone from data_prepare/.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_prepare'))
from dataload import LibriPhraseDataset
from qwen_audio_encoder import QwenAudioEncoder

a = LibriPhraseDataset()
# a = LibriPhraseDataset(types = 'easy')
# a = LibriPhraseDataset(types = 'hard')
data = a.get_data()

anchor_wav = data['anchor_wav'].values
anchor_text = data['anchor_text'].values
comparison_text = data['comparison_text'].values
comparison_wav = data['comparison_wav'].values
label = data['label'].values
datatype = data['type'].values

y_scores = []
text_tmp = []
new_file_scp = []
# device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qwen_enc = QwenAudioEncoder(model_id="Qwen/Qwen2-Audio-7B", device=device, max_frames=100)

# save_dir = '/train20/intern/temporary/kwli2/udkws/interfea/whisperenc/100h/'
save_dir = '/train20/intern/temporary/kwli2/udkws/interfea/whisperenc/360h/4word/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for i in tqdm(range(len(anchor_wav))):
    audio_path_an = anchor_wav[i]
    audio, _ = librosa.load(audio_path_an, sr=16000, mono=True)
    anchor_feature = qwen_enc.encode(audio)  # [1, <=100, 1280] [t, c]

    audio_path_com = comparison_wav[i]
    audio_com, _ = librosa.load(audio_path_com, sr=16000, mono=True)
    com_feature = qwen_enc.encode(audio_com)  # [1, <=100, 1280] [t, c]

    data_dict={'anchor_speech':anchor_feature, 'comparison_speech':com_feature, 'anchor_text':anchor_text[i], 'comparison_text':comparison_text[i], 'type':datatype[i], 'label':label[i], 'anchor_path':anchor_wav[i], 'comparison_path':comparison_wav[i]}
    # data_dict={'clean_speech':a_feature, 'noisy_speech':noisy_speech, 'type':datatype[i], 'label':label[i], 'speech_labels':wav_label[i], 'text_labels':text[i]}
    name = anchor_wav[i].split('/')[-1].split('.')[0] + '_' + comparison_wav[i].split('/')[-1].split('.')[0]
    save_path = save_dir + name + '.npy'
    np.save(save_path, data_dict)

    new_file_scp.append(save_path) 

with open('/train20/intern/permanent/kwli2/udkws/tmp1/npyfile/testeasy_all_360h_4word.scp', 'w') as output:
    output.writelines(it + '\n' for it in new_file_scp)
