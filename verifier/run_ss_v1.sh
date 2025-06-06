#!/bin/bash
set -e

json_path='/home/t-yangrui/code/Screenspot_eval/json_data/new_prompt_7B/screenspot_all_preds_Original.json'
exp_name='Actor-7b-warmup-fixprompt-bon_score_verifierultracpt6000_crop500'
verifier_path='microsoft/GUI-Actor-Verifier-2B'
screenspot_dataset_path="data/ss-eval/ScreenSpot"
logdir='results_v1'

verifier_method='score'
# verifier_method='best_one'
export CUDA_VISIBLE_DEVICES=0


python eval_ss_with_verifier.py  \
    --screenspot_imgs ${screenspot_dataset_path}'/images'  \
    --screenspot_test ${screenspot_dataset_path}  \
    --task "all" \
    --language "en" \
    --gt_type "positive" \
    --log_path "${logdir}/${exp_name}_${checkpoint}_ssv1.json" \
    --inst_style "instruction" \
    --verifier_method ${verifier_method} \
    --verifier_path ${verifier_path}  \
    --json_prediction  ${json_path} 

 