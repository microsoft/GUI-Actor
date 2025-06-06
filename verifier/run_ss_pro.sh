# !/bin/bash
set -e

json_path='Screenspot_eval/json_data/7B_full_qwen2vl/final_eval/screenspot-Pro_all_preds_StandardResize.json'
exp_name='Actor-7b-fixprompt-bon_score_verifier'
verifier_path='microsoft/GUI-Actor-Verifier-2B'
screenspot_dataset_path='/datadisk/data/ss-eval/ScreenSpot-Pro'
logdir='results_pro'

verifier_method='score' 
# verifier_method='best_one'
export CUDA_VISIBLE_DEVICES=0


python eval_ss_with_verifier.py  \
    --screenspot_imgs ${screenspot_dataset_path}'/images'  \
    --screenspot_test ${screenspot_dataset_path}'/annotations'  \
    --task "all" \
    --language "en" \
    --gt_type "positive" \
    --log_path "${logdir}/${exp_name}_${checkpoint}_sspro.json" \
    --inst_style "instruction" \
    --verifier_method ${verifier_method} \
    --verifier_path ${verifier_path} \
    --json_prediction  ${json_path} 

