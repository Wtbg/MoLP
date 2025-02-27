train_ratio_list=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
num_splits=5
for train_ratio in "${train_ratio_list[@]}"
do
    echo "Inference for train ratio $train_ratio ↓"
    for i in $(seq 1 $num_splits)
    do
        checkpoint_dir=checkpoints/v4_repeat/ratio_$train_ratio/split_$i
        mkdir -p inference/v4_repeat/ratio_$train_ratio/split_$i
        result_dir=inference/v4_repeat/ratio_$train_ratio/split_$i/results_v4.json
        python inference/inference.py --checkpoint_path $checkpoint_dir --results_path $result_dir
    done 
done