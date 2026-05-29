import math, os, re, sys
from pathlib import Path
import random
import numpy as np
import pandas as pd
from scipy.io import wavfile
import time
import wave
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import librosa
import math
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))


class LipReading2Dataset(Dataset):
    def __init__(self,
                 rank, 
                 LOG,
                 train=True,
                 types='both', 
                 scp_path='/train8/asrkws/shicheng2/Data/Multimodal_KWS/LRS2/dataset_list',
                 train_csv='train',
                 test_csv='eval_inset,eval_outset',
                 prob_addNoise=0.5,
                 snr_list=[3,6,9],
                 seed=42):

        phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                  'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B',   'CH',
                                  'D',   'DH',  'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                  'EY2', 'F',   'G',   'HH',  'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                  'JH',  'K',   'L',   'M',   'N',   'NG',  'OW0', 'OW1', 'OW2', 'OY0',
                                  'OY1', 'OY2', 'P',   'R',   'S',   'SH',  'T',   'TH',  'UH0', 'UH1',
                                  'UH2', 'UW',  'UW0', 'UW1', 'UW2', 'V',   'W',   'Y',   'Z',   'ZH',
                                  ' ']
        
        self.p2idx = {p: idx for idx, p in enumerate(phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(phonemes)}

        self.train = train
        self.types = types
        self.scp_path = scp_path
        self.train_csv = train_csv.split(',')
        self.test_csv = test_csv.split(',')
        self.prob_addNoise = prob_addNoise
        self.seed = seed

        self.files_scp = []
        if self.types == 'both':
            pass
        elif self.types == 'inset':
            self.test_csv = [self.test_csv[0]]
        elif self.types == 'outset':
            self.test_csv = [self.test_csv[1]]

        for db in self.train_csv if self.train else self.test_csv:
            csv_list = [str(x) for x in Path(self.scp_path).rglob('shuf_' + db + '.scp')]
            self.train_csv
            self.files_scp_tmp = []
            
            for n_word in csv_list:
                if rank == 0:
                    if self.train:
                        if LOG != None:
                            LOG.log(">> Loading train_list: {} ".format(n_word))
                    else:
                        if LOG != None:
                            LOG.log(">> Loading test_list: {} {}".format(n_word, str(snr_list[0]) + 'dB' if snr_list != None else 'clean'))
                self.files_scp_tmp = []
        
                with open(n_word) as f:
                    lines = f.readlines()
                self.files_scp_tmp = [line.strip() for line in lines]
                self.files_scp = self.files_scp + self.files_scp_tmp
            
        self.sample_rate = 16000
        self.maxlen_text = 40
        self.maxlen_vide = 50
        self.maxlen_audi = 100
        self.rng = np.random.default_rng(self.seed)
        self.amp_noise_list  = list(range(200, 1000, 10))
        self.rng.shuffle(self.amp_noise_list)
        self.snr_list = snr_list

    def __len__(self):
        return len(self.files_scp)

    def __read_audio__(self, wav_path):
        with wave.open(wav_path, 'rb') as wf:
            sample_width = wf.getsampwidth()  # 采样宽度（字节）
            frame_rate = wf.getframerate()    # 采样率
            n_frames = wf.getnframes()        # 总帧数
            audio_data = wf.readframes(n_frames)
        
        if sample_width == 2:  # 如果是 16-bit PCM
            dtype = np.int16
        elif sample_width == 4:  # 如果是 32-bit PCM
            dtype = np.int32
        else:
            raise ValueError("不支持的样本宽度: {}".format(sample_width))
        
        audio_array = np.frombuffer(audio_data, dtype=dtype)

        return sample_width, frame_rate, audio_array


    def __getitem__(self, idx):

        data_path = self.files_scp[idx]
        feature_data = np.load(data_path, allow_pickle=True).item()

        ## Text feature
        anc_phn_list = feature_data['anc_phn_list']
        anc_text_fea = feature_data['anc_text_fea']
        com_phn_list = feature_data['com_phn_list']
        com_text_fea = feature_data['com_text_fea']

        if not torch.is_tensor(anc_text_fea):
            anc_text_fea = torch.from_numpy(np.asarray(anc_text_fea))
        if not torch.is_tensor(com_text_fea):
            com_text_fea = torch.from_numpy(np.asarray(com_text_fea))

        anc_phn_list = torch.tensor([self.p2idx[t] for t in anc_phn_list])
        com_phn_list = torch.tensor([self.p2idx[t] for t in com_phn_list])

        masktmp_anc = anc_phn_list != 71
        masktmp_com = com_phn_list != 71

        anc_phn_list = anc_phn_list[anc_phn_list!=71]
        com_phn_list = com_phn_list[com_phn_list!=71]
        
        len_anc = len(anc_phn_list)
        len_com = len(com_phn_list)

        anc_phn_list = F.pad(anc_phn_list, (0, self.maxlen_text-len_anc), "constant", 0)
        com_phn_list = F.pad(com_phn_list, (0, self.maxlen_text-len_com), "constant", 0)

        anc_text_fea = torch.masked_select(anc_text_fea, masktmp_anc.bool().unsqueeze(-1)).view(-1,256)
        com_text_fea = torch.masked_select(com_text_fea, masktmp_com.bool().unsqueeze(-1)).view(-1,256)

        anc_text_fea = F.pad(anc_text_fea, (0, 0, 0, self.maxlen_text-len_anc), "constant", 0).float()
        com_text_fea = F.pad(com_text_fea, (0, 0, 0, self.maxlen_text-len_com), "constant", 0).float()

        anc_text_mask = torch.ones((self.maxlen_text, 1))
        anc_text_mask[len_anc:, :] = 0
        com_text_mask = torch.ones((self.maxlen_text, 1))
        com_text_mask[len_com:, :] = 0


        ## video feature
        anc_vide_fea = np.load(feature_data['anc_vide_fea_path'])
        if anc_vide_fea.shape[0] == 1:
            anc_vide_fea = anc_vide_fea.squeeze(0)
        anc_vide_fea = torch.tensor(anc_vide_fea).cpu().detach()
        com_vide_fea = np.load(feature_data['com_vide_fea_path'])
        if com_vide_fea.shape[0] == 1:
            com_vide_fea = com_vide_fea.squeeze(0)
        com_vide_fea = torch.tensor(com_vide_fea).cpu().detach()

        
        len_anc_lip = anc_vide_fea.shape[0]
        len_com_lip = com_vide_fea.shape[0]

        anc_vide_mask = torch.ones((self.maxlen_vide, 1)) # 16khz 音频数据，/320对应分辨率20ms，也即self.audio_fps = 50
        anc_vide_mask[len_anc_lip:, :] = 0
        com_vide_mask = torch.ones((self.maxlen_vide, 1))
        com_vide_mask[len_com_lip:, :] = 0

        anc_vide_mask = torch.ones((self.maxlen_vide, 1)) # 16khz 音频数据，/320对应分辨率20ms，也即self.audio_fps = 50
        anc_vide_mask[len_anc_lip:, :] = 0
        com_vide_mask = torch.ones((self.maxlen_vide, 1))
        com_vide_mask[len_com_lip:, :] = 0

        # 视频embed统一补成2s长
        T_anc, C_anc = anc_vide_fea.shape
        if T_anc < self.maxlen_vide:
            tmp_anc_vide_fea = torch.zeros((self.maxlen_vide, C_anc), device=anc_vide_fea.device)
            tmp_anc_vide_fea[:T_anc, :] = anc_vide_fea
        else:
            tmp_anc_vide_fea = anc_vide_fea[:self.maxlen_vide, :]
        anc_vide_fea = tmp_anc_vide_fea

        T_com, C_com = com_vide_fea.shape
        if T_com < 50:
            tmp_com_vide_fea = torch.zeros((50, C_com), device=com_vide_fea.device)
            tmp_com_vide_fea[:T_com, :] = com_vide_fea
        else:
            tmp_com_vide_fea = com_vide_fea[:50, :]
        com_vide_fea = tmp_com_vide_fea


        ## audio feature
        anc_audi_fea = np.load(feature_data['anc_audi_fea_path']).squeeze(0)
        anc_audi_fea = torch.tensor(anc_audi_fea).cpu().detach()

        if self.train:
            if self.rng.random(1)[0] <= self.prob_addNoise:
                snr = self.rng.choice(self.snr_list, replace=False)
                com_audi_fea_path = feature_data['com_audi_fea_path'].replace('/wav/', f'/wav_{snr}db/')
                com_audi_fea = np.load(com_audi_fea_path).squeeze(0)
            else:
                com_audi_fea = np.load(feature_data['com_audi_fea_path']).squeeze(0)
        else:
            if self.snr_list == None:                                                           # clean场景
                com_audi_fea = np.load(feature_data['com_audi_fea_path']).squeeze(0)
            else:                                                                               # noisy场景
                snr = self.snr_list[0]
                for type_string in ['/testall_jiwai/wav/', '/testhard/wav/', '/testeasy/wav/']:
                    if type_string in feature_data['com_audi_fea_path']:
                        com_audi_fea_path = feature_data['com_audi_fea_path'].replace(type_string, f'/test/{snr}db/')
                com_audi_fea = np.load(com_audi_fea_path).squeeze(0)
        
        com_audi_fea = torch.tensor(com_audi_fea).cpu().detach()

        _, _, anc_wav = self.__read_audio__(feature_data['anc_wav_path'])
        _, _, com_wav = self.__read_audio__(feature_data['com_wav_path'])
        anc_wav = anc_wav[:self.maxlen_audi*20*160] # B, T <= 32000
        com_wav = com_wav[:self.maxlen_audi*20*160]
        
        ## fa
        anc_fa_path = feature_data['anc_wav_path'].replace('/wav/', '/fa_data/').replace('_wav', '').replace('.wav', '.TextGrid')
        com_fa_path = feature_data['com_wav_path'].replace('/wav/', '/fa_data/').replace('_wav', '').replace('.wav', '.TextGrid')

        ##  audio
        k_anc = math.ceil(len(anc_wav) / 320) # 16khz 音频数据，/320对应分辨率20ms，也即self.audio_fps = 50
        k_com = math.ceil(len(com_wav) / 320)
        anc_audi_mask = torch.ones((self.maxlen_audi, 1)) # 最长100帧，也即2s音频，训练测试数据基本在此限制内
        anc_audi_mask[k_anc:, :] = 0
        com_audi_mask = torch.ones((self.maxlen_audi, 1))
        com_audi_mask[k_com:, :] = 0

        ## label
        label = feature_data['label']
        if self.train:
            clean_com_audi_fea = np.load(feature_data['com_audi_fea_path']).squeeze(0)
            clean_com_audi_fea = torch.tensor(clean_com_audi_fea).cpu().detach()

            if 'type' in feature_data.keys():
                datatype = feature_data['type']
            else:
                if feature_data['anc_text'] == feature_data['com_text']:
                    datatype = "diffspk_positive"
            if datatype == "diffspk_positive" or datatype == "positive":  # 正例，注册与查询文本完全一致，来自不同说话人
                label_3 = 1
            elif datatype == 'diffspk_easyneg' or datatype == 'easy_negative': # 负例，注册与查询文本较不一致，来自不同说话人
                label_3 = 0
            elif datatype == 'diffspk_hardneg' or datatype == 'hard_negative': # 负例，注册与查询文本相似但不一致，来自不同说话人
                label_3 = 2
            else:
                print("Wrong datatype:", datatype)

            return anc_audi_fea, anc_audi_mask, com_audi_fea, com_audi_mask, anc_phn_list, anc_text_fea, anc_text_mask, com_phn_list, com_text_fea, com_text_mask, anc_vide_fea, anc_vide_mask, com_vide_fea, com_vide_mask, anc_fa_path, com_fa_path, label, clean_com_audi_fea, label_3

        else:
            return anc_audi_fea, anc_audi_mask, com_audi_fea, com_audi_mask, anc_phn_list, anc_text_fea, anc_text_mask, com_phn_list, com_text_fea, com_text_mask, anc_vide_fea, anc_vide_mask, com_vide_fea, com_vide_mask, anc_fa_path, com_fa_path, label


class CollaterTrain():
    def __init__(self, train = True):
        self.train = train

    def __call__(self, sample_batch):
        
        data = {}
        meta = {}
        fa_path = {}

        anc_audi_fea  = [x[0]  for x in sample_batch]
        anc_audi_mask = [x[1]  for x in sample_batch]
        com_audi_fea  = [x[2]  for x in sample_batch]
        com_audi_mask = [x[3]  for x in sample_batch]
        anc_phn_list  = [x[4]  for x in sample_batch]
        anc_text_fea  = [x[5]  for x in sample_batch]
        anc_text_mask = [x[6]  for x in sample_batch]
        com_phn_list  = [x[7]  for x in sample_batch]
        com_text_fea  = [x[8]  for x in sample_batch]
        com_text_mask = [x[9]  for x in sample_batch]
        anc_vide_fea  = [x[10] for x in sample_batch]
        anc_vide_mask = [x[11] for x in sample_batch]
        com_vide_fea  = [x[12] for x in sample_batch]
        com_vide_mask = [x[13] for x in sample_batch]

        anc_fa_path   = [x[14] for x in sample_batch]
        com_fa_path   = [x[15] for x in sample_batch]
        label         = [x[16] for x in sample_batch]


        anc_audi_fea  = pad_sequence(anc_audi_fea,  batch_first=True, padding_value=0.0) 
        anc_audi_mask = pad_sequence(anc_audi_mask, batch_first=True, padding_value=0.0) 
        com_audi_fea  = pad_sequence(com_audi_fea,  batch_first=True, padding_value=0.0)
        com_audi_mask = pad_sequence(com_audi_mask, batch_first=True, padding_value=0.0) 
        anc_phn_list  = pad_sequence(anc_phn_list,  batch_first=True, padding_value=0.0) 
        anc_text_fea  = pad_sequence(anc_text_fea,  batch_first=True, padding_value=0.0) 
        anc_text_mask = pad_sequence(anc_text_mask, batch_first=True, padding_value=0.0) 
        com_phn_list  = pad_sequence(com_phn_list,  batch_first=True, padding_value=0.0) 
        com_text_fea  = pad_sequence(com_text_fea,  batch_first=True, padding_value=0.0) 
        com_text_mask = pad_sequence(com_text_mask, batch_first=True, padding_value=0.0) 
        anc_vide_fea  = pad_sequence(anc_vide_fea,  batch_first=True, padding_value=0.0)
        anc_vide_mask  = pad_sequence(anc_vide_mask,  batch_first=True, padding_value=0.0)
        com_vide_fea  = pad_sequence(com_vide_fea,  batch_first=True, padding_value=0.0) 
        com_vide_mask  = pad_sequence(com_vide_mask,  batch_first=True, padding_value=0.0)

        label         = torch.tensor(label)
        
        data['anc_audi_fea']   = anc_audi_fea
        data['anc_audi_mask']  = anc_audi_mask
        data['com_audi_fea']   = com_audi_fea
        data['com_audi_mask']  = com_audi_mask
        data['anc_phn_list']   = anc_phn_list
        data['anc_text_fea']   = anc_text_fea
        data['anc_text_mask']  = anc_text_mask
        data['com_phn_list']   = com_phn_list
        data['com_text_fea']   = com_text_fea
        data['com_text_mask']  = com_text_mask
        data['anc_vide_fea']   = anc_vide_fea
        data['anc_vide_mask']  = anc_vide_mask
        data['com_vide_fea']   = com_vide_fea
        data['com_vide_mask']  = com_vide_mask

        fa_path['anc_fa_path'] = anc_fa_path
        fa_path['com_fa_path'] = com_fa_path

        meta['label'] = label

        if self.train:
            clean_com_audi_fea = [x[17] for x in sample_batch]
            clean_com_audi_fea  = pad_sequence(clean_com_audi_fea,  batch_first=True, padding_value=0.0)
            data['clean_com_audi_fea']   = clean_com_audi_fea

            label_3   = [x[18] for x in sample_batch]
            label_3   = torch.tensor(label_3)
            meta['label_3'] = label_3

        return data, meta, fa_path

def worker_init_fn_train(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    num_workers = worker_info.num_workers
    worker_idx = worker_info.id

def worker_init_fn_eval(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset


def get_train_dataloader(args, LOG, type, snr_list):
    train_dataset = LipReading2Dataset(args.gpu_global_rank, LOG, train=True, types=type, scp_path=args.datalist_dir, train_csv=args.train_csv, 
                                       prob_addNoise=args.prob_addNoise, snr_list=snr_list, seed=args.seed)

    sampler_train = DistributedSampler(train_dataset, num_replicas=torch.cuda.device_count(), rank=torch.cuda.current_device()) 

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        num_workers=2,
        collate_fn=CollaterTrain(train = True),
        shuffle=False,
        sampler=sampler_train,
        multiprocessing_context="spawn"
    )
    return train_dataloader

def get_val_dataloader(args, LOG, type, snr_list):
    val_dataset = LipReading2Dataset(args.gpu_global_rank, LOG, train=False, types=type, scp_path=args.datalist_dir, test_csv=args.eval_csv,
                                     prob_addNoise=args.prob_addNoise, snr_list=snr_list, seed=42)

    # sampler_train = DistributedSampler(val_dataset, num_replicas=torch.cuda.device_count(), rank=torch.cuda.current_device()) 

    val_dataloader = DataLoader(
        val_dataset,
        batch_size = args.batch_size,
        num_workers=2, 
        collate_fn=CollaterTrain(train = False),
        shuffle=False,
        # sampler = sampler_train,
        multiprocessing_context="spawn"
    )
    return val_dataloader


def get_test_dataloader(args, type, snr_list):
    val_dataset = LipReading2Dataset(rank=0, LOG=None, train=False, types=type, scp_path=args.datalist_dir, test_csv=args.eval_csv,
                                     prob_addNoise=args.prob_addNoise, snr_list=snr_list, seed=42)

    # sampler_train = DistributedSampler(val_dataset, num_replicas=torch.cuda.device_count(), rank=torch.cuda.current_device()) 

    val_dataloader = DataLoader(
        val_dataset,
        batch_size = args.batch_size,
        num_workers=1, 
        collate_fn=CollaterTrain(train = False),
        shuffle=False,
        # sampler = sampler_train,
        multiprocessing_context="spawn"
    )
    return val_dataloader
