train_ratio_list=(0.1 0.2 0.3 0.4 0.5)
num_splits=5
for train_ratio in "${train_ratio_list[@]}"
do
    echo "Training MoE for train ratio $train_ratio ↓"
    for i in $(seq 1 $num_splits)
    do
        echo "Training MoE for train ratio $train_ratio for split $i"
        log_dir=training/v4_repeat_3part_0.1/ratio_$train_ratio/split_$i
        checkpoint_dir=checkpoints/v4_repeat_3part_0.1/ratio_$train_ratio/split_$i
        train_data_dir=data/split/v4_repeat_3part_0.1/ratio_$train_ratio/split_$i/train/embedding
        val_data_dir=data/split/v4_repeat_3part_0.1/ratio_$train_ratio/split_$i/val/embedding
        train_label_dir=data/split/v4_repeat_3part_0.1/ratio_$train_ratio/split_$i/train/label
        val_label_dir=data/split/v4_repeat_3part_0.1/ratio_$train_ratio/split_$i/val/label
        python main_unconf.py --log_dir $log_dir --checkpoint_dir $checkpoint_dir --train_data_dir $train_data_dir --val_data_dir $val_data_dir --train_label_dir $train_label_dir --val_label_dir $val_label_dir
    done
done