import argparse
from pathlib import Path
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file_list_train", type=str, default="training/split/v4/train_set_0.8.txt")
    parser.add_argument("--input_label_file_list_train", type=str, default="training/split/v4/train_set_label_0.8.txt")
    parser.add_argument("--input_data_file_list_val", type=str, default="training/split/v4/val_set_0.8.txt")
    parser.add_argument("--input_label_file_list_val", type=str, default="training/split/v4/val_set_label_0.8.txt")
    parser.add_argument("--output_dir", type=str, default="data/v4/ratio_0.8")
    
    args = parser.parse_args()
    
    output_dir_train = Path(args.output_dir) / "train"
    output_dir_val = Path(args.output_dir) / "val"
    
    output_dir_train_embedding = output_dir_train / "embedding"
    output_dir_train_label = output_dir_train / "label"
    
    output_dir_val_embedding = output_dir_val / "embedding"
    output_dir_val_label = output_dir_val / "label"
    
    output_dir_train_embedding.mkdir(parents=True, exist_ok=True)
    output_dir_train_label.mkdir(parents=True, exist_ok=True)
    output_dir_val_embedding.mkdir(parents=True, exist_ok=True)
    output_dir_val_label.mkdir(parents=True, exist_ok=True)
    with open(args.input_data_file_list_train, "r") as f:
        data_files = f.readlines()
    with open(args.input_label_file_list_train, "r") as f:
        label_files = f.readlines()
    for data_file in data_files:
        data_file = data_file.strip()
        shutil.copy(data_file, output_dir_train_embedding)
    for label_file in label_files:
        label_file = label_file.strip()
        shutil.copy(label_file, output_dir_train_label)
    with open(args.input_data_file_list_val, "r") as f:
        data_files = f.readlines()
    with open(args.input_label_file_list_val, "r") as f:
        label_files = f.readlines()
    for data_file in data_files:
        data_file = data_file.strip()
        shutil.copy(data_file, output_dir_val_embedding)
    for label_file in label_files:
        label_file = label_file.strip()
        shutil.copy(label_file, output_dir_val_label)

        