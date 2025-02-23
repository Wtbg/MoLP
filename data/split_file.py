import argparse
from pathlib import Path
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_file", type=str, default="training/split/train_set_label.txt") 
    parser.add_argument("--output_dir", type=str, default="data/train/processed/labels")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.id_file, "r") as f:
        ids = f.readlines()
    for file in ids:
        file = file.strip()
        file_path = Path(file)
        # copy file to output_dir
        shutil.copy(file_path, output_dir/file_path.name)
    print("Split file successfully.")