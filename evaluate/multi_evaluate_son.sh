#!/bin/bash
model_list=(llavanextvideo7b longva7b chat-scene leo video_3d_llm llava-3d)
train_ratio=$1
id_set_arg="training/split/v4/ratio_${train_ratio}/val_set_${train_ratio}.txt"
output_dir_arg="evaluate/v4/ratio_${train_ratio}"
echo "Evaluate MoE! Train ratio: $train_ratio"
python evaluate/evaluate_unconf.py --id_set $id_set_arg --output_dir $output_dir_arg 
echo "Evaluate Each Model! Train ratio: $train_ratio"
for model in "${model_list[@]}"
do
    echo "Evaluate $model! Train ratio: $train_ratio"
    python evaluate/evaluate_unconf.py --id_set $id_set_arg --output_dir $output_dir_arg --evaluate_benchmark_name $model --if_evaluate_benchmark True
done