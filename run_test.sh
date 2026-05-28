source /home3/asrkws/shicheng2/bashrc_multimodal_kws
CUDA_VISIBLE_DEVICES=0 python test.py \
--bgn_epoch 0 \
--end_epoch 15 \
--batch_size 1 \
--test_snrs 5,0,-5,-10 \
--network 'TVA_KWS_PLCL_AVmask' \
--datalist_dir '/local/scratch/linna/MISP/MISP_baseline/MISP-QEKS/dataset_list' \
--eval_csv 'test_inset,test_outset' \
--prob_addNoise 1.0 \
--model_path '/local/scratch/linna/MISP/MISP_baseline/MISP-QEKS/train/model/' \
--out_dir './test_epoch10/' \
--maxlen_text 40 \
--maxlen_vide 50 \
--maxlen_audi 100 \
# --eval_csv 'eval_inset,eval_outset' \
# --eval_csv 'test_inset,test_outset' \
# --eval_csv 'eval_debug_inset,eval_debug_outset'
