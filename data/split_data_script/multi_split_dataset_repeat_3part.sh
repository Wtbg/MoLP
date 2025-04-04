train_ratio_list=(0.1 0.05)
num_splits=5
for train_ratio in "${train_ratio_list[@]}"
do
    echo "Splitting dataset for train ratio $train_ratio"
    output_dir=training/split/v4_repeat_3part/ratio_$train_ratio
    python data/split_dataset_unconf_repeat_3part.py --output_dir $output_dir --train_ratio $train_ratio --num_splits $num_splits --val_ratio $train_ratio 
done
