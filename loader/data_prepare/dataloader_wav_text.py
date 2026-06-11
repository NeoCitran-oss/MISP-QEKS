from pathlib import Path
import sys, os
import pandas as pd
from torch.utils.data import Dataset
import torch

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import difflib
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

class LibriPhraseDataset():
    def __init__(self,
                #  wav_dir = '/train20/intern/temporary/kwli2/lib2/test',
                 wav_dir = '/train20/intern/temporary/kwli2/lib2',
                 noise_dir = '/train20/intern/permanent/kwli2/phonmachnet/dataset/noise',
                #  csv_dir = '/train20/intern/permanent/kwli2/phonmachnet/PhonMatchNet-main/dataext/datatest',
                 csv_dir = '/train20/intern/permanent/kwli2/phonmachnet/LibriPhrase-main/datafin',
                 train_csv=['all_100h', 'all_360h'],
                 test_csv=['all_360h', ], # all_100h all(500h) all_360h
                 types='both',  # easy, hard
                 features='g2p_embed',  # phoneme, g2p_embed, both ...
                 train=False,
                 pkl=None,
                 edit_dist=False,
                 datalen = 0):
        self.wav_dir = wav_dir
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.train = train
        self.csv_dir = csv_dir
        self.types = types
        self.data = pd.DataFrame(columns=['anchor_text', 'anchor_wav', 'comparison_text', 'comparison_wav', 'duration', 'label', 'type'])
    def get_data(self):
        for db in self.train_csv if self.train else self.test_csv:
            csv_list = [str(x) for x in Path(self.csv_dir).rglob('*' + db + '*4word*')]
            for n_word in csv_list:
                print(">> processing : {} ".format(n_word))
                df = pd.read_csv(n_word)
                # df = df[:20]
                anc_pos = df[['anchor_text', 'anchor', 'anchor_text', 'anchor_dur']]
                anc_neg = df[['anchor_text', 'anchor', 'comparison_text', 'anchor_dur', 'target', 'type']]
                com_pos = df[['comparison_text', 'comparison', 'comparison_text', 'comparison_dur']]
                com_neg = df[['anchor_text', 'anchor', 'comparison_text', 'comparison',  'comparison_dur', 'target', 'type']]
                anc_pos.columns = ['wav_label', 'anchor', 'anchor_text', 'anchor_dur']
                com_pos.columns = ['wav_label', 'comparison', 'comparison_text', 'comparison_dur']
                anc_pos['label'] = 1
                anc_pos['type'] = df['type']
                com_pos['label'] = 1
                com_pos['type'] = df['type']

                # self.data = self.data.append(anc_pos.rename(columns={y: x for x, y in zip(self.data.columns, anc_pos.columns)}), ignore_index=True)
                # self.data = self.data.append(anc_neg.rename(columns={y: x for x, y in zip(self.data.columns, anc_neg.columns)}), ignore_index=True)
                # self.data = self.data.append(com_pos.rename(columns={y: x for x, y in zip(self.data.columns, com_pos.columns)}), ignore_index=True)
                self.data = self.data.append(com_neg.rename(columns={y: x for x, y in zip(self.data.columns, com_neg.columns)}), ignore_index=True)
                
            self.data['anchor_wav'] = self.data['anchor_wav'].apply(lambda x: os.path.join(self.wav_dir, x))
            self.data['comparison_wav'] = self.data['comparison_wav'].apply(lambda x: os.path.join(self.wav_dir, x))
        # self.data = self.data.sort_values(by='duration').reset_index(drop=True)
        if self.types == 'both':
            pass
        elif self.types == 'easy':
            self.dataeasy = self.data.loc[self.data['type'] == 'diffspk_easyneg']
            self.dataeasy = self.dataeasy.append(self.data.loc[self.data['type'] == 'diffspk_positive'])
            self.data = self.dataeasy
            
        elif self.types == 'hard':
            self.datahard = self.data.loc[self.data['type'] == 'diffspk_hardneg']
            self.datahard = self.datahard.append(self.data.loc[self.data['type'] == 'diffspk_positive'])
            self.data = self.datahard

        self.data.to_csv('output4.csv', index=False)

        # self.wav_list = self.data['wav'].values
        # self.idx_list = self.data['text'].values
        # self.lab_list = self.data['label'].values
        return self.data



   
# a = LibriPhraseDataset()
# data = a.get_data()