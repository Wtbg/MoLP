#!/bin/bash
train_ratio_list=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
for train_ratio in "${train_ratio_list[@]}"
do
    bash evaluate/multi_evaluate_son.sh $train_ratio
done
