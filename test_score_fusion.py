"""
Evaluate XEQ-Matcher with score fusion instead of feature concatenation.

Feature fusion (default test.py):
    concat(pair_embeddings) -> fc_tva_va -> 1 logit

Score fusion (this script):
    fc_t_v(embd_t_v), fc_t_a(embd_t_a), ... -> weighted fusion -> 1 logit
"""
import argparse
import logging
import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tabulate import tabulate

from loader.dataloader_test import get_test_dataloader
from model.score_fusion import PAIR_NAMES, parse_fusion_weights
from model.tva_model import TVA_KWS_PLCL_AVmask


def compute_eer(label, pred):
    fpr, tpr, _ = roc_curve(label, pred)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2


def validation(model, dataloader, fusion_weights, fusion_method):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for data, meta, fa_path in dataloader:
            for key in data:
                data[key] = data[key].cuda()

            fused_logit, component_logits = model(
                data,
                meta,
                fa_path,
                return_mode='scores',
                fusion_weights=fusion_weights,
                fusion_method=fusion_method,
            )
            predictions.extend(torch.sigmoid(fused_logit).cpu().numpy()[:, 0])
            labels.extend(meta['label'].cpu().numpy())

    auc = round(roc_auc_score(labels, predictions), 4)
    eer = round(compute_eer(labels, predictions), 4)
    model.train()
    return auc, eer


def test(args):
    fusion_weights = parse_fusion_weights(args.fusion_weights)

    with torch.cuda.device(0):
        torch.manual_seed(args.net_init_seed)
        torch.cuda.manual_seed_all(args.net_init_seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

        model = TVA_KWS_PLCL_AVmask().cuda()
        model.init_score_fusion_heads_from_checkpoint()

        inset_loaders = {}
        outset_loaders = {}

        inset_loaders['clean'] = get_test_dataloader(args, type='inset', snr_list=None)
        outset_loaders['clean'] = get_test_dataloader(args, type='outset', snr_list=None)

        test_snr_list = [int(i) for i in args.test_snrs.split(',')] if args.test_snrs else [5, 0, -5, -10]
        for snr in test_snr_list:
            key = f'{snr} dB'
            inset_loaders[key] = get_test_dataloader(args, type='inset', snr_list=[snr])
            outset_loaders[key] = get_test_dataloader(args, type='outset', snr_list=[snr])

        logging.info('Score fusion method: %s', args.fusion_method)
        logging.info('Score fusion weights: %s', fusion_weights)
        logging.info('Testing SNR items: clean & %s', test_snr_list)

        for epoch in range(args.bgn_epoch, args.end_epoch + 1):
            ckpt_path = os.path.join(args.model_path, f'epoch{epoch}.pth')
            state_dict = torch.load(ckpt_path, map_location='cpu')
            match = model.load_state_dict(state_dict['state_dict'], strict=False)
            model.init_score_fusion_heads_from_checkpoint()
            model = model.cuda()
            logging.info('Loaded %s, %s', ckpt_path, match)

            headers = ['SNR', 'AUC_score_fusion', 'EER_score_fusion']
            inset_rows = []
            outset_rows = []

            for test_item in inset_loaders:
                logging.info('Testing inset: %s', test_item)
                inset_auc, inset_eer = validation(
                    model, inset_loaders[test_item], fusion_weights, args.fusion_method
                )
                outset_auc, outset_eer = validation(
                    model, outset_loaders[test_item], fusion_weights, args.fusion_method
                )
                inset_rows.append([test_item, inset_auc, inset_eer])
                outset_rows.append([test_item, outset_auc, outset_eer])

            logging.info(
                'Inset Results:\n%s',
                tabulate(inset_rows, headers=headers, tablefmt='fancy_grid'),
            )
            logging.info(
                'Outset Results:\n%s',
                tabulate(outset_rows, headers=headers, tablefmt='fancy_grid'),
            )

        logging.info('*' * 150)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MISP-QEKS score fusion evaluation')
    parser.add_argument('--test_snrs', type=str, default='5,0,-5,-10')
    parser.add_argument('--eval_csv', type=str, default='eval_inset,eval_outset')
    parser.add_argument(
        '--datalist_dir',
        type=str,
        default='/local/scratch/linna/MISP/MISP_baseline/MISP-QEKS/data_list',
    )
    parser.add_argument('--prob_addNoise', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--bgn_epoch', type=int, default=9)
    parser.add_argument('--end_epoch', type=int, default=9)
    parser.add_argument('--seed', type=int, default=27863875)
    parser.add_argument('--net_init_seed', type=int, default=27863875)
    parser.add_argument('--out_dir', type=str, default='./test_score_fusion')
    parser.add_argument(
        '--model_path',
        type=str,
        default='/local/scratch/linna/MISP/MISP_data/MISP-QEKS/train/model/',
    )
    parser.add_argument('--maxlen_text', type=int, default=40)
    parser.add_argument('--maxlen_vide', type=int, default=50)
    parser.add_argument('--maxlen_audi', type=int, default=100)
    parser.add_argument(
        '--fusion_method',
        type=str,
        default='weighted_mean',
        choices=['weighted_mean', 'max', 'product'],
        help='How to combine pairwise match scores.',
    )
    parser.add_argument(
        '--fusion_weights',
        type=str,
        default='',
        help=(
            'Pairwise score weights. Empty = equal 1.0 for all. '
            f'Order: {",".join(PAIR_NAMES)} or names like t_v=1,t_a=2,...'
        ),
    )

    args = parser.parse_args()
    args.Mem_bank = None
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.out_dir, 'test_score_fusion.log'),
        level=logging.DEBUG,
        format='[%(asctime)s][%(levelname)s]  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    test(args)
    print('finished !!!')
