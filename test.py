import argparse
import copy
import os
import glob 
import shutil
from loader.dataloader_test import get_test_dataloader

from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from time import time
import logging

import os
import sys
import random
import math
import numpy as np
import time
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist
from torch.backends import cudnn
import torch.nn as nn
import socket
import copy
import glob
from collections import OrderedDict
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from library.layers import *
from library.optim import Lookahead
from library.utils.message import Message, LogType, get_log_header
from library.train_helper import get_module, printer

from tabulate import tabulate
from model.lipreading import video_encoder


from pytorch_optimizer import Ranger
from model.wbmodel import *
from model.cosafind import *
from model.atmodelphone import *
from model.tva_model import TVA_KWS_PLCL_AVmask
from model.clap_phn import ContrastiveLoss_mask
import pickle
sys.setrecursionlimit(10000) # zqs


def compute_eer(label, pred):
    fpr, tpr, threshold = roc_curve(label, pred)
    fnr = 1 - tpr

    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    eer = (eer_1 + eer_2) / 2
    return eer

class EER(nn.Module):
    def __init__(self):
        super(EER, self).__init__()
        self.score = 0.0
        self.count = 0.0

    def forward(self, y_true, y_pred):
        label_np = y_true.flatten()  # Convert to numpy array
        pred_np = y_pred.flatten()  # Convert to numpy array

        eer_value = compute_eer(label_np, pred_np)

        self.score += eer_value
        self.count += 1

        return torch.tensor(self.score / self.count)


def validation(model, Val_dataloader):
    model.eval()
    pred_tva_va = []

    label_list = []

    with torch.no_grad():
        for data, meta, fa_path in Val_dataloader:

            for key in data:
                data[key] = data[key].cuda()

            prob_tva_va = model(data, meta, fa_path, modality='tva_va')[0]

            outputs_val_tva_va = torch.sigmoid(prob_tva_va).cpu().numpy()

            pred_tva_va.extend(outputs_val_tva_va[:,0])

            label = meta['label'].cuda()
            label = label.unsqueeze(-1).float()
            label_list.extend(list(label[:,0].cpu().numpy()))

        # Calculating AUC
        roc_auc_tva_va = round(roc_auc_score(label_list, pred_tva_va), 4)    # TVA

        # Calculating EER
        eer_tva_va = round(compute_eer(label_list, pred_tva_va), 4)
    
    model.train()
    AUC = [roc_auc_tva_va]
    EER = [eer_tva_va]

    return AUC, EER

def test(args):

    with torch.cuda.device(0):

        # 初始化模型
        torch.manual_seed(args.net_init_seed)
        torch.cuda.manual_seed_all(args.net_init_seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

        network_class  = globals()[args.network]
        model = network_class()
        model = model.cuda()
        
        Inset_Test_dataloaders = {}
        Outset_Test_dataloaders = {}

        Inset_Test_dataloaders['clean']  = get_test_dataloader(args, type='inset',  snr_list=None)
        Outset_Test_dataloaders['clean'] = get_test_dataloader(args, type='outset', snr_list=None)

        if args.test_snrs:
            test_snr_list = [int(i) for i in args.test_snrs.split(',')]
        else:
            test_snr_list = [5, 0, -5, -10]
        for snr in test_snr_list:
            test_item = f'{snr} dB'
            Inset_Test_dataloaders[test_item]  = get_test_dataloader(args, type='inset',  snr_list=[snr])
            Outset_Test_dataloaders[test_item] = get_test_dataloader(args, type='outset', snr_list=[snr])
        
        logging.info("Testing data summary: {}".format(len(Inset_Test_dataloaders['clean'])))
        logging.info("Testing snr items: clean & {}".format(test_snr_list))

        for epoch in range(args.bgn_epoch, args.end_epoch + 1):

            path = args.model_path + 'epoch' + str(epoch) + '.pth'
            state_dict = torch.load(path, map_location='cpu')
            match = model.load_state_dict(state_dict["state_dict"], strict=False)
            model = model.cuda()
            logging.info("load {}, {}".format(path, match))

            table_eval_inset  = []
            table_eval_outset = []

            headers = ["SNR", "AUC_TVA_VA", "EER_TVA_VA"]


            for test_item in Inset_Test_dataloaders.keys():
                logging.info("Testing: {} Scenario".format(test_item))

                row_inset   = [test_item]
                row_outset  = [test_item]

                Inset_AUC,  Inset_EER  = validation(model, Inset_Test_dataloaders[test_item])
                Outset_AUC, Outset_EER = validation(model, Outset_Test_dataloaders[test_item])

                row_inset.extend(Inset_AUC)
                row_inset.extend(Inset_EER)

                row_outset.extend(Outset_AUC)
                row_outset.extend(Outset_EER)

                table_eval_inset.append(row_inset)
                table_eval_outset.append(row_outset)

            tabulate_eval_inset  = tabulate(table_eval_inset, headers=headers, tablefmt="fancy_grid")
            tabulate_eval_outset = tabulate(table_eval_outset, headers=headers, tablefmt="fancy_grid")

            logging.info(f"Inset Results: \n{tabulate_eval_inset}")
            logging.info(f"Outset Results:\n{tabulate_eval_outset}")

        logging.info("*"*150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_snrs',     type=str,   default='0,5,10')
    parser.add_argument('--eval_csv',      type=str,   default='eval_inset,eval_outset')
    parser.add_argument('--datalist_dir',  type=str,   default='/local/scratch/linna/MISP/MISP_baseline/MISP-QEKS/npy/eval')
    parser.add_argument('--prob_addNoise', type=float, default=1.0)
    parser.add_argument('--batch_size',     type=int,   default=16, help='Max sentence number used for training and testing')
    parser.add_argument('--bgn_epoch',     type=int,   default=2)
    parser.add_argument('--end_epoch',     type=int,   default=3)
    parser.add_argument('--network',       type=str,   default='TVA_KWS_PLCL_AVmask')
    parser.add_argument('--seed',          type=int,   default=27863875)
    parser.add_argument('--net_init_seed', type=int,   default=27863875)

    parser.add_argument('--out_dir',       type=str,   default='./test')
    parser.add_argument('--model_path',    type=str,   default='./train/cfg1/model/')

    parser.add_argument('--maxlen_text',   type=int,   default=40)
    parser.add_argument('--maxlen_vide',   type=int,   default=50)
    parser.add_argument('--maxlen_audi',   type=int,   default=100)

    args = parser.parse_args()
    args.Mem_bank=None

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.out_dir, 'test.log'),
        level=logging.DEBUG,
        format='[%(asctime)s][%(levelname)s]  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


    test(args)

    print('finished !!!')


