source /home3/asrkws/shicheng2/bashrc_multimodal_kws
python train.py \
--lr 0.01 \
--use_bmuf \
--epochs 10 \
--batch_size 64 \
--train_snrs 3,6,9 \
--test_snrs 5,0,-5,-10 \
--optimizer SGD \
--network 'TVA_KWS_PLCL_AVmask' \
--datalist_dir '/local/scratch/linna/MISP/MISP_baseline/MISP-QEKS/data_list' \
--train_csv 'train' \
--eval_csv 'eval_inset,eval_outset' \
--prob_addNoise 0.6 \
--lr_half_epochs 2,4,6,8 \
--out_dir './train/model/' \
--log_path './train/0_train.log' \
--display 40 \
--maxlen_text 40 \
--maxlen_vide 50 \
--maxlen_audi 100 \
# --train_csv 'train' \
# --train_csv 'checkout' \
