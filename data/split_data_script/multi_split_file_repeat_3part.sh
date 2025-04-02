train_ratio_list=(0.1 0.05)
num_splits=5
for train_ratio in "${train_ratio_list[@]}"
do
    for i in $(seq 1 $num_splits)
    do
        echo "Splitting dataset for train ratio $train_ratio for split $i"
        input_data_file_list_train=training/split/v4_repeat_3part/ratio_$train_ratio/split_$i/train_set_$i.txt
        input_label_file_list_train=training/split/v4_repeat_3part/ratio_$train_ratio/split_$i/train_set_label_$i.txt
        input_data_file_list_val=training/split/v4_repeat_3part/ratio_$train_ratio/split_$i/val_set_$i.txt
        input_label_file_list_val=training/split/v4_repeat_3part/ratio_$train_ratio/split_$i/val_set_label_$i.txt
        output_dir=data/split/v4_repeat_3part/ratio_$train_ratio/split_$i
        python data/split_file_unconf.py --input_data_file_list_train $input_data_file_list_train --input_label_file_list_train $input_label_file_list_train --input_data_file_list_val $input_data_file_list_val --input_label_file_list_val $input_label_file_list_val --output_dir $output_dir
    done
done