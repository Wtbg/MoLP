#!/bin/bash
train_ratio_list=(0.1 0.2 0.3 0.4 0.5)
num_splits=5
for train_ratio in "${train_ratio_list[@]}"
do
    echo "Evaluating MoE for train ratio $train_ratio ↓"
    for i in $(seq 1 $num_splits)
    do
        echo "Evaluating MoE for train ratio $train_ratio for split $i"
        bash evaluate/repeat_bash/multi_evaluate_son_3part_0.1.sh $train_ratio $i
    done 
done
