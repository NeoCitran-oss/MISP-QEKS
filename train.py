import argparse
import copy
import os
import glob 
import shutil
from loader.dataloader import get_train_dataloader, get_val_dataloader

from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from time import time

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

class BMUF():
    def __init__(self, model, args, dist):
        momentums = []
        global_models = []
        for param in model.parameters():
            temp = torch.zeros_like(param, requires_grad=False)
            temp.copy_(param.data)
            global_models.append(temp)
            momentums.append(torch.zeros_like(param, requires_grad=False))
        momentums = torch.nn.utils.parameters_to_vector(momentums)
        global_models = torch.nn.utils.parameters_to_vector(global_models)

        self.momentums = momentums
        self.global_models = global_models
        self.bmuf_alpha = args.bmuf_alpha
        self.bmuf_bm = args.bmuf_bm
        self.bmuf_blr = args.bmuf_blr
        self.dist = dist

    def update(self, model):
        self.__update_param(model)

    def __update_param(self, model):
        size = float(self.dist.get_world_size())
        v = torch.nn.utils.parameters_to_vector(model.parameters())
        avg = v.detach().clone()
        self.dist.all_reduce(avg.data, op=dist.ReduceOp.SUM)
        avg.data /= size
        update = self.bmuf_bm * self.momentums + self.global_models
        grad = avg - update
        self.momentums.copy_(self.bmuf_blr * grad +
                             self.bmuf_bm * self.momentums)
        self.global_models.copy_(self.global_models + self.momentums)
        update = self.bmuf_bm * self.momentums + self.global_models
        v.data.copy_(v.detach() - self.bmuf_alpha * (v.detach() - update))
        torch.nn.utils.vector_to_parameters(v, model.parameters())

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
    if len(Val_dataloader.dataset) == 0:
        # No scp/pairs for this split (e.g. eval_outset not prepared) — skip.
        return [float("nan")], [float("nan")]
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


def train(args, LOG):
    if args.gpu_world_size > 1:
        # gloo: NCCL needs NVML to enumerate every physical GPU and this host has
        # broken GPUs 6/7 (nvmlDeviceGetHandleByIndex fails) -> ncclSystemError.
        dist.init_process_group(args.dist_backend, init_method=args.initmethod, world_size=args.gpu_world_size, rank=args.gpu_global_rank)
        logtype=LogType.SYNC
    else:
        logtype=LogType.COMMON

    with torch.cuda.device(args.gpu_local_rank):
        torch.manual_seed(args.net_init_seed)
        torch.cuda.manual_seed_all(args.net_init_seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

        LOG.log("rank: {}".format(args.gpu_local_rank))

        if args.train_snrs:
            train_snr_list = [int(i) for i in args.train_snrs.split(',')]
        train_dataloader = get_train_dataloader(args, LOG, type='both', snr_list=train_snr_list)
        if args.gpu_global_rank == 0:
            LOG.log("train: {}".format(len(train_dataloader)))

        network_class  = globals()[args.network]
        model = network_class()

        LOG.log("model: {}".format(model.__class__.__name__),
                    rank=args.gpu_global_rank, logtype=logtype, syncid=-1, syncheader="load previous model")
        model = model.cuda()
        LOG.log("gpu: {}".format(args.gpu_global_rank),
                    rank=args.gpu_global_rank, logtype=logtype, syncid=-1, syncheader="GPU Infomation")
        LOG.log("dataloader len: {}".format(len(train_dataloader)),
                    rank=args.gpu_global_rank, logtype=logtype, syncid=-1, syncheader="Dataset Summary")


        model = model.cuda()

        if getattr(args, 'resume_from', None):
            ckpt = torch.load(args.resume_from, map_location='cpu')
            sd = ckpt.get('state_dict', ckpt)
            # checkpoints saved from a DDP-wrapped model carry a "module." prefix
            sd = OrderedDict((k[len('module.'):] if k.startswith('module.') else k, v)
                             for k, v in sd.items())
            model.load_state_dict(sd)
            LOG.log("Resumed weights from {} (start_epoch={})".format(args.resume_from, args.start_epoch),
                    rank=args.gpu_global_rank, logtype=logtype, syncid=-1, syncheader="resume")

        # 初始化优化器
        bmuf = None
        if args.gpu_world_size > 1:
            if args.use_bmuf:
                bmuf = BMUF(model, args, dist)
            else:
                # find_unused_parameters: some branches (e.g. modality-specific heads)
                # do not contribute to the tva_va loss, so their params get no grad.
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu_local_rank], find_unused_parameters=True)
        if args.optimizer == 'ADAM':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'RANGER':
            optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.0, nesterov=True)

        if args.lr_half_epochs:
            lr_half_epochs = [int(i) for i in args.lr_half_epochs.split(',')]
        else:
            lr_half_epochs = [100000000000]

        scheduler = MultiStepLR(optimizer, milestones=lr_half_epochs, gamma=0.5)
        # fast-forward LR schedule to the resume position
        for _ in range(args.start_epoch):
            scheduler.step()

        if args.test_snrs:
            test_snr_list = [int(i) for i in args.test_snrs.split(',')]


        Inset_Val_dataloaders  = {}
        Outset_Val_dataloaders = {}
        if args.gpu_global_rank == 0:
            LOG.log("Logging Inset_Val_dataloaders")
        Inset_Val_dataloaders['clean']  = get_val_dataloader(args, LOG, type='inset', snr_list=None)
        for snr in test_snr_list:
            test_item = f'{snr}dB'
            Inset_Val_dataloaders[test_item]  = get_val_dataloader(args, LOG, type='inset', snr_list=[snr])

        if args.gpu_global_rank == 0:
            LOG.log("Logging Outset_Val_dataloaders")
        Outset_Val_dataloaders['clean'] = get_val_dataloader(args, LOG, type='outset', snr_list=None)
        for snr in test_snr_list:
            test_item = f'{snr}dB'
            Outset_Val_dataloaders[test_item] = get_val_dataloader(args, LOG, type='outset', snr_list=[snr])
       

        if args.gpu_global_rank == 0:
            LOG.log("Eval: {}".format(len(Inset_Val_dataloaders['clean'])))
            LOG.log("Training seed: {}".format(args.seed))

        for epoch in range(args.start_epoch, args.end_epoch):

            sum_of_last_displayed_loss = 0
            sum_of_last_displayed_loss_t_a   = 0
            sum_of_last_displayed_loss_a_a   = 0
            sum_of_last_displayed_loss_ta_a  = 0
            sum_of_last_displayed_loss_ta_va = 0
            sum_of_last_displayed_loss_tva_a = 0
            sum_of_last_displayed_loss_tva_va = 0

            sum_of_last_displayed_claploss_tv = 0
            sum_of_last_displayed_claploss_ta = 0
            sum_of_last_displayed_claploss_vv = 0
            sum_of_last_displayed_claploss_va = 0
            sum_of_last_displayed_claploss_av = 0
            sum_of_last_displayed_claploss_aa = 0

            sum_of_last_displayed_glm_loss = 0
            sum_of_last_displayed_denoised_loss = 0

            model.train()

            for i, data_element in enumerate(train_dataloader):

                data, meta, fa_path = data_element

                for key in data:
                    data[key] = data[key].cuda()
                for key in meta:
                    meta[key] = meta[key].cuda()

                optimizer.zero_grad()
                i = i + 1

                prdic = model(data, meta, fa_path, modality='tva_va')

                if prdic != 0:
                    prob, claploss_tv, claploss_ta, claploss_vv, claploss_va, claploss_av, claploss_aa, loss_t_a, loss_a_a, loss_ta_a, loss_ta_va, loss_tva_a, loss_tva_va, glm_loss, denoised_loss = prdic
                else:
                    continue

                loss = loss_t_a + loss_a_a + loss_ta_a + loss_ta_va + loss_tva_a + loss_tva_va + claploss_tv * 0.1 + claploss_ta * 0.1 + claploss_vv * 0.1 + claploss_va * 0.1 + claploss_av * 0.1 + claploss_aa * 0.1 + glm_loss * 0.5 + denoised_loss * 0.01

                sum_of_last_displayed_loss += loss.item()
                sum_of_last_displayed_loss_t_a    += loss_t_a.item()
                sum_of_last_displayed_loss_a_a    += loss_a_a.item()
                sum_of_last_displayed_loss_ta_a   += loss_ta_a.item()
                sum_of_last_displayed_loss_ta_va  += loss_ta_va.item()
                sum_of_last_displayed_loss_tva_a  += loss_tva_a.item()
                sum_of_last_displayed_loss_tva_va += loss_tva_va.item()

                sum_of_last_displayed_claploss_tv += claploss_tv.item()
                sum_of_last_displayed_claploss_ta += claploss_ta.item()
                sum_of_last_displayed_claploss_vv += claploss_vv.item()
                sum_of_last_displayed_claploss_va += claploss_va.item()
                sum_of_last_displayed_claploss_av += claploss_av.item()
                sum_of_last_displayed_claploss_aa += claploss_aa.item()

                sum_of_last_displayed_glm_loss += glm_loss.item()
                sum_of_last_displayed_denoised_loss += denoised_loss.item()

                loss.backward()

                optimizer.step()

                if i % args.display == 0 and i > 0:
                    # Total Loss 
                    displayed_loss = sum_of_last_displayed_loss / args.display
                    
                    # Attention Loss
                    displayed_loss_t_a = sum_of_last_displayed_loss_t_a / args.display
                    displayed_loss_a_a = sum_of_last_displayed_loss_a_a / args.display
                    displayed_loss_ta_a = sum_of_last_displayed_loss_ta_a / args.display
                    displayed_loss_ta_va = sum_of_last_displayed_loss_ta_va / args.display
                    displayed_loss_tva_a = sum_of_last_displayed_loss_tva_a / args.display
                    displayed_loss_tva_va = sum_of_last_displayed_loss_tva_va / args.display

                    # Phone-Level Contrastive Learning Loss
                    displayed_claploss_tv = sum_of_last_displayed_claploss_tv / args.display
                    displayed_claploss_ta = sum_of_last_displayed_claploss_ta / args.display
                    displayed_claploss_vv = sum_of_last_displayed_claploss_vv / args.display
                    displayed_claploss_va = sum_of_last_displayed_claploss_va / args.display
                    displayed_claploss_av = sum_of_last_displayed_claploss_av / args.display
                    displayed_claploss_aa = sum_of_last_displayed_claploss_aa / args.display

                    # Denoise Loss
                    displayed_glm_loss = sum_of_last_displayed_glm_loss / args.display
                    displayed_denoised_loss = sum_of_last_displayed_denoised_loss / args.display

                    loss_dict = {
                        'loss': displayed_loss,
                        'loss_t_a': displayed_loss_t_a,
                        'loss_a_a': displayed_loss_a_a,
                        'loss_ta_a': displayed_loss_ta_a,
                        'loss_ta_va': displayed_loss_ta_va,
                        'loss_tva_a': displayed_loss_tva_a,
                        'loss_tva_va': displayed_loss_tva_va,
                        'claploss_tv': displayed_claploss_tv,
                        'claploss_ta': displayed_claploss_ta,
                        'claploss_vv': displayed_claploss_vv,
                        'claploss_va': displayed_claploss_va,
                        'claploss_av': displayed_claploss_av,
                        'claploss_aa': displayed_claploss_aa,
                        'glm_loss': displayed_glm_loss,
                        'denoised_loss': displayed_denoised_loss,
                    }
                    formatted_log = ", ".join([f"{key}:{value:.3f}" for key, value in loss_dict.items()])
                    learning_rate = scheduler.get_last_lr()[0]
                    LOG.log(f"[TRAIN]: epoch:{epoch} part:{i} lr: {learning_rate} gpu:{args.gpu_global_rank}  [LOSS]: {formatted_log}", rank=args.gpu_global_rank, logtype=logtype, syncid=i, syncheader="epoch{0} part{1}".format(epoch, i))

                    sum_of_last_displayed_loss = 0
                    sum_of_last_displayed_loss_t_a   = 0
                    sum_of_last_displayed_loss_a_a   = 0
                    sum_of_last_displayed_loss_ta_a  = 0
                    sum_of_last_displayed_loss_ta_va = 0
                    sum_of_last_displayed_loss_tva_a = 0
                    sum_of_last_displayed_loss_tva_va = 0

                    sum_of_last_displayed_claploss_tv = 0
                    sum_of_last_displayed_claploss_ta = 0
                    sum_of_last_displayed_claploss_vv = 0
                    sum_of_last_displayed_claploss_va = 0
                    sum_of_last_displayed_claploss_av = 0
                    sum_of_last_displayed_claploss_aa = 0
            
                    sum_of_last_displayed_glm_loss = 0
                    sum_of_last_displayed_denoised_loss = 0


            if args.gpu_world_size > 1:
                dist.barrier()
            
            if args.gpu_global_rank == 0:
                model_path = os.path.join(args.out_dir, "epoch{}.pth".format(epoch))
                model_save = model
                state_dict = model_save.cpu().state_dict()

                torch.save({"state_dict": state_dict}, model_path)

                LOG.log("save model path.. {}\n".format(model_path))

                model = model.cuda()
                
                table_eval_inset  = []
                table_eval_outset = []

                headers = ["SNR", "AUC_TVA_VA", "EER_TVA_VA"]
                
                for test_item in Inset_Val_dataloaders.keys():
                    if args.gpu_global_rank == 0:
                        LOG.log("Testing: {} Scenario".format(test_item))
                    
                    row_inset   = [test_item]
                    row_outset  = [test_item]

                    Inset_AUC,  Inset_EER  = validation(model, Inset_Val_dataloaders[test_item])
                    Outset_AUC, Outset_EER = validation(model, Outset_Val_dataloaders[test_item])

                    row_inset.extend(Inset_AUC)
                    row_inset.extend(Inset_EER)

                    row_outset.extend(Outset_AUC)
                    row_outset.extend(Outset_EER)

                    table_eval_inset.append(row_inset)
                    table_eval_outset.append(row_outset)


                tabulate_eval_inset  = tabulate(table_eval_inset, headers=headers, tablefmt="fancy_grid")
                tabulate_eval_outset = tabulate(table_eval_outset, headers=headers, tablefmt="fancy_grid")

                LOG.log(f"Inset  Results:\n{tabulate_eval_inset}\n")
                LOG.log(f"Outset Results:\n{tabulate_eval_outset}")

            scheduler.step()

            if args.gpu_world_size > 1:
                dist.barrier()

def train_warper(args, message_queue, LOG):
    status = printer(train)(args, LOG)
    message_queue.put(status)

def start_train(args):
    cudnn.benchmark = False
    cudnn.enabled = True
    if "MASTER_ADDR" in os.environ:
        master_addr = socket.gethostbyname(os.environ["MASTER_ADDR"])
        master_port = int(os.environ["MASTER_PORT"])
        log_port = master_port+1
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["RANK"])
    else:
        master_addr = socket.gethostbyname("localhost")
        master_port = random.randint(20000, 30000)
        log_port = master_port+1
        args.world_size = 1
        args.rank = 0
    if args.gpu_num is None:
        args.gpu_num = torch.cuda.device_count()
    args.gpu_world_size = args.world_size * args.gpu_num
    print(f'Using {args.gpu_world_size} gpus in total')

    args.net_init_seed = args.seed if args.seed else random.randint(0,10000)
    if args.gpu_world_size > 1:
        args.initmethod = "tcp://{0}:{1}".format(master_addr, master_port)
        if args.use_bmuf:
            if args.gpu_world_size == 4:
                args.bmuf_alpha = 0.75
                args.bmuf_bm = 0.75
                args.bmuf_blr = 1.0
            elif args.gpu_world_size == 2:
                args.bmuf_alpha = 0.75
                args.bmuf_bm = 0.75
                args.bmuf_blr = 1.0
            elif args.gpu_world_size == 8:
                args.bmuf_alpha = 1.0
                args.bmuf_bm = 0.9
                args.bmuf_blr = 1.0
            elif args.gpu_world_size == 12:
                args.bmuf_alpha = 1.0
                args.bmuf_bm = 0.92
                args.bmuf_blr = 1.0
            elif args.gpu_world_size == 16:
                args.bmuf_alpha = 1.0
                args.bmuf_bm = 0.94
                args.bmuf_blr = 1.0
            elif args.gpu_world_size == 20:
                args.bmuf_alpha = 1.0
                args.bmuf_bm = 0.954
                args.bmuf_blr = 1.0
            elif args.gpu_world_size == 24:
                args.bmuf_alpha = 1.0
                args.bmuf_bm = 0.965
                args.bmuf_blr = 1.0
            elif args.gpu_world_size == 28:
                args.bmuf_alpha = 1.0
                args.bmuf_bm = 0.97
                args.bmuf_blr = 1.0
            elif args.gpu_world_size == 32:
                args.bmuf_alpha = 1.0
                args.bmuf_bm = 0.972
                args.bmuf_blr = 1.0
            else:
                raise Exception("gpu_world_size {}, bmuf train failed".format(args.gpu_world_size))
    
    message = Message(master_addr, log_port, args.world_size)
    if args.rank == 0:
        message.start_server()
    else:
        message.start_client()
    if args.rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        message.set_logpath(args.log_path, args.gpu_world_size)
    LOG = message.getlog()

    log = ''
    log += get_log_header("Args Summary")
    for k in list(vars(args).keys()):
        log += "{}: {}\n".format(k, vars(args)[k])
    if args.rank == 0:
        LOG.log(log)

    log = ''
    log += get_log_header("Dataset Summary")
        

    resume_from = None
    resume_start_epoch = 0
    if args.resume:
        ckpts = glob.glob(os.path.join(args.out_dir, 'epoch*.pth'))
        if ckpts:
            def _epoch_num(p):
                return int(os.path.basename(p)[len('epoch'):-len('.pth')])
            resume_from = max(ckpts, key=_epoch_num)
            resume_start_epoch = _epoch_num(resume_from) + 1
            if args.rank == 0:
                LOG.log("--resume: latest checkpoint {} -> continuing at epoch {}".format(
                    resume_from, resume_start_epoch))
        elif args.rank == 0:
            LOG.log("--resume: no epoch*.pth in {} - starting fresh".format(args.out_dir))

    ctx = mp.get_context('spawn')
    message_queue = ctx.Queue()

    process = []
    for gpu_local_rank in range(args.gpu_num):
        subargs = copy.deepcopy(args)
        subargs.gpu_local_rank = gpu_local_rank
        subargs.gpu_global_rank = subargs.rank * subargs.gpu_num + gpu_local_rank
        subargs.start_epoch = resume_start_epoch
        subargs.resume_from = resume_from
        subargs.end_epoch = subargs.epochs
        p = ctx.Process(target=train_warper, args=(subargs, message_queue, LOG,))
        p.start()
        process.append(p)
    status = 1
    for i in range(args.gpu_num):
        status = status * message_queue.get()
        if status == 0:
            LOG.log("train process failed")
            message.shutdown()
            break
    for p in process:
        p.join()
    if status == 1:
        LOG.log("train accomplished")
    else:
        raise Exception("train process failed")
    message.wait()
    message.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_at',  type=str, default=None, help='Path of pretrain_at model, auto detected')
    parser.add_argument('--pretrain_aa',  type=str, default=None, help='Path of pretrain_aa model, auto detected')

    parser.add_argument('--train_snrs',   type=str, default='3,6,9',  help='Index of training snr values')
    parser.add_argument('--test_snrs',    type=str, default='0,5,10', help='Index of testing snr values')

    parser.add_argument('--train_csv',    type=str, default='train')
    parser.add_argument('--eval_csv',     type=str, default='eval_inset,eval_outset')
    parser.add_argument('--datalist_dir', type=str, default='/local/scratch/linna/MISP/MISP_baseline/MISP-QEKS/data_list', help='Path where the scp stored')

    parser.add_argument('--prob_addNoise',  type=float, default=0.5, help='probability of adding noise')

    parser.add_argument('--optimizer',      type=str,   default='SGD', help='SGD/ADAM')
    parser.add_argument('--lr',             type=float, default=0.001, help='Learning rate for train')
    parser.add_argument('--epochs',         type=int,   default=8, help='Number of training epochs')
    parser.add_argument('--batch_size',     type=int,   default=16, help='Max sentence number used for training and testing')
    parser.add_argument('--network',        type=str,   default='TVA_KWS_PLCL_AVmask', help='Deep learning model architecture used')
    parser.add_argument('--lr_half_epochs', type=str,   default='2,3,4,5,6,7,8,9', help='Index of training epoch to half learning rate, eg: 3,4,5,6,7')
    parser.add_argument('--seed',           type=int,   default=27863875, help='Seed for init pytorch network and dataloader')
    parser.add_argument('--net_init_seed',  type=int,   default=27863875, help='Seed for init pytorch network, especially for multigpu init')

    parser.add_argument('--display',  type=int, default=100, help='Number of training iteration to display')
    parser.add_argument('--out_dir',  type=str, default='./train', help='Path to where the output model will be')
    parser.add_argument('--log_path', type=str, default='./train/0_train.log', help='Path to where the output log will be')

    parser.add_argument('--use_bmuf', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume from the latest epochN.pth in --out_dir (continues at epoch N+1, restores LR schedule)')
    parser.add_argument('--dist_backend', type=str, default='gloo', choices=['nccl', 'gloo'], help='DDP backend; gloo avoids NVML (broken GPUs 6/7 on this host)')
    parser.add_argument('--gpu_num',  type=int, default=None, help='Gpu number, auto get by torch.cuda.device_count() if not set')

    parser.add_argument('--maxlen_text', type=int, default=40, help='maxlen_text')
    parser.add_argument('--maxlen_vide', type=int, default=50, help='maxlen_text')
    parser.add_argument('--maxlen_audi', type=int, default=100, help='maxlen_audi')

    args = parser.parse_args()
    
    start_train(args)

    print('finished !!!')
