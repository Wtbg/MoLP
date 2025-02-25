import argparse
import json
from pathlib import Path
import pdb
import random
from utils.config import load_config

def split_dataset(full_dataset, train_ratio):
    type_sets = {}
    for item in full_dataset:
        type_info = item["type_info"]
        id = item["id"]
        if type_info not in type_sets:
            type_sets[type_info] = []
        type_sets[type_info].append(id)
    train_set = []
    val_set = []
    for type_info, ids in type_sets.items():
        random.shuffle(ids)
        split_idx = int(len(ids) * train_ratio)
        print(f"Type {type_info}: {len(ids)} samples, {split_idx} for training, {len(ids) - split_idx} for validation.")
        train_ids = ids[:split_idx]
        val_ids = ids[split_idx:]
        for id in train_ids:
            train_set.append(id)
        for id in val_ids:
            val_set.append(id)
    
    return train_set, val_set






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = load_config("configs/default.yaml")
    parser.add_argument("--full_dataset_path", type=str, default="backgroundata/modelresults/merged_v4.json")
    parser.add_argument("--output_dir", type=str, default="training/split/v4")
    args = parser.parse_args()
    with open(args.full_dataset_path, "r") as f:
        data = json.load(f)
    train_ratio = float(config.data.train_ratio)
    print(f"Splitting dataset with train ratio {train_ratio}.")
    train_set, val_set = split_dataset(data, train_ratio)
    output_dir = Path(args.output_dir)
    
    train_filename = output_dir / f"train_set_{train_ratio}.txt"
    val_filename = output_dir / f"val_set_{train_ratio}.txt"

    with open(train_filename, "w") as f:
        for id in train_set:
            f.write(f"data/v4/processed/embeddings/{str(id).zfill(8)}.pt\n")

    with open(val_filename, "w") as f:
        for id in val_set:
            f.write(f"data/v4/processed/embeddings/{str(id).zfill(8)}.pt\n")
            
    train_label_filename = output_dir / f"train_set_label_{train_ratio}.txt"
    val_label_filename = output_dir / f"val_set_label_{train_ratio}.txt"
    
    with open(train_label_filename, "w") as f:
        for id in train_set:
            f.write(f"data/v4/processed/labels/{str(id).zfill(8)}.pt\n")
    
    with open(val_label_filename, "w") as f:
        for id in val_set:
            f.write(f"data/v4/processed/labels/{str(id).zfill(8)}.pt\n")
    
    print("Split dataset successfully.")


