CUDA_VISIBLE_DEVICES=0 python test_score_fusion.py \
--bgn_epoch 9 \
--end_epoch 9 \
--batch_size 1 \
--test_snrs 5,0,-5,-10 \
--datalist_dir '/local/scratch/linna/MISP/MISP_baseline/MISP-QEKS/data_list' \
--eval_csv 'eval_inset,eval_outset' \
--prob_addNoise 1.0 \
--model_path '/local/scratch/linna/MISP/MISP_data/MISP-QEKS/train/model/' \
--out_dir './test_score_fusion/' \
--fusion_method weighted_mean \
--fusion_weights '1,1,1,1,1,1' \
--maxlen_text 40 \
--maxlen_vide 50 \
--maxlen_audi 100
# Pair order for --fusion_weights: t_v,t_a,v_v,v_a,a_v,a_a
# Examples:
#   --fusion_method max
#   --fusion_weights 't_v=1,t_a=2,v_v=1,v_a=1,a_v=2,a_a=1'
