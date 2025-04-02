train_ratio_list=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
num_splits=5
for train_ratio in "${train_ratio_list[@]}"
do
    echo "Splitting dataset for train ratio $train_ratio"
    output_dir=training/split/v4_repeat/ratio_$train_ratio
    python data/split_dataset_unconf_repeat.py --output_dir $output_dir --train_ratio $train_ratio --num_splits $num_splits
done
