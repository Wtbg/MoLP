#!/bin/bash
model_list=(llavanextvideo7b longva7b chat-scene leo video_3d_llm llava-3d)
train_ratio=$1
split_idx=$2
id_set_arg="training/split/v4_repeat_3part/ratio_${train_ratio}/split_${split_idx}/test_set_${split_idx}.txt"
output_dir_arg="evaluate/v4_repeat_3part/ratio_${train_ratio}/split_${split_idx}"
inference_results=inference/v4_repeat_3part/ratio_$train_ratio/split_$split_idx/results_v4.json
echo "Evaluate MoE! Train ratio: $train_ratio for split $split_idx"
python evaluate/evaluate_unconf.py --id_set $id_set_arg --output_dir $output_dir_arg --inference_results $inference_results
echo "Evaluate Each Model! Train ratio: $train_ratio for split $split_idx"
for model in "${model_list[@]}"
do
    echo "Evaluate $model! Train ratio: $train_ratio"
    python evaluate/evaluate_unconf.py --id_set $id_set_arg --output_dir $output_dir_arg --evaluate_benchmark_name $model --if_evaluate_benchmark True --inference_results $inference_results
done