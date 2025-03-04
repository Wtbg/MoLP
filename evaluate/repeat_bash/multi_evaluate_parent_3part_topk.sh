#!/bin/bash
train_ratio_list=(0.1)
num_splits=5
topk=$1
for train_ratio in "${train_ratio_list[@]}"
do
    echo "Evaluating MoE for train ratio $train_ratio ↓"
    for i in $(seq 1 $num_splits)
    do
        echo "Evaluating MoE for train ratio $train_ratio for split $i"
        bash evaluate/repeat_bash/multi_evaluate_son_3part_topk.sh $train_ratio $i $topk
    done 
done
