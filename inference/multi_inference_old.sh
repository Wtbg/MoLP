train_ratio_list=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
num_splits=5
for train_ratio in "${train_ratio_list[@]}"
do
    echo "Inference for train ratio $train_ratio ↓"
    checkpoint_dir=training/v4/checkpoints/ratio_$train_ratio
    mkdir -p inference/v4/ratio_$train_ratio
    result_dir=inference/v4/ratio_$train_ratio/results_v4.json
    python inference/inference.py --checkpoint_path $checkpoint_dir --results_path $result_dir
done