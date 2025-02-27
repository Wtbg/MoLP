import argparse
import json
import pdb
import sys
import numpy as np
sys.path.append("/sda/kongming/3d-cake/script/MoLP")
from utils.scripts.generate_label import calculate_mra_single, calculate_mra_array, mean_relative_accuracy

MODELS = {  "llavanextvideo7b":0,
            "longva7b":1,
            "chat-scene":2,
            "leo":3,
            "video_3d_llm":4,
            "llava-3d":5
}

BEST_MODEL_FOR_EACH_CATE = {
    0: {"best_model":"chat-scene"}, # obj_count
    1: {"best_model":"leo"}, # odd_even
    2: {"best_model":"llavanextvideo7b"}, # size
    3: {"best_model":"leo"}, # front_back
    4: {"best_model":"chat-scene"}, # left_right
    5: {"best_model":"llavanextvideo7b"}, # nearest
}


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    return data


def get_key_from_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None


def cal_mra_cly(pred, target):
    mra = mean_relative_accuracy(pred, target, .5, .95, .05)
    return mra


def evaluate_based_on_labels(merged_file, ratio_val_file, output_file, best_model_list):
    # 读取相关数据
    data = read_json_file(merged_file)
    val_data = read_txt_file(ratio_val_file)
    # 所需变量
    val_ids = set()
    final_scores = {
        0: {"scores":0.00, "count":0}, 
        1: {"scores":0.00, "count":0},
        2: {"scores":0.00, "count":0},
        3: {"scores":0.00, "count":0},
        4: {"scores":0.00, "count":0},
        5: {"scores":0.00, "count":0}
    }
    llava_3d_scores = {"scores":0.00, "count":0, "avg":0.00}
    results = {}

    for line in val_data:
        id = line.split('.')[0].split('/')[-1].lstrip('0').strip() # 处理得到数据id
        val_ids.add(int(id))
    
    
    for item in data:
        id = item['id']
        if id in val_ids: # 只对验证集进行评估
            category = item['type_info']
            best_model = best_model_list[category]["best_model"] # 选择该label表现上最优的模型
            pred = item['models'][best_model]
            target = item['answer']
            if category in [0,2]:
                try:
                    pred_float = float(pred)
                except ValueError:
                    pred_float = 0.00
                mra = cal_mra_cly(pred_float, float(target))
                final_scores[category]["scores"] += mra
                final_scores[category]["count"] += 1
            elif category in [1, 3, 4, 5]:
                final_scores[category]["count"] += 1
                if pred == target:
                    final_scores[category]["scores"] += 1
            else:
                print("Error: category not found")
                print(item)
                exit()
    
    results["type0_mra"] = final_scores[0]["scores"] / final_scores[0]["count"]
    results["type2_mra"] = final_scores[2]["scores"] / final_scores[2]["count"]
    results["type1_accuracy"] = final_scores[1]["scores"] / final_scores[1]["count"]
    results["type3_accuracy"] = final_scores[3]["scores"] / final_scores[3]["count"]
    results["type4_accuracy"] = final_scores[4]["scores"] / final_scores[4]["count"]
    results["type5_accuracy"] = final_scores[5]["scores"] / final_scores[5]["count"]
    results['avergae_score'] = (results["type0_mra"] + results["type2_mra"] + results["type1_accuracy"] + results["type3_accuracy"] + results["type4_accuracy"] + results["type5_accuracy"]) / 6

    print(results)

    write_json_file(results, output_file)


def eval_on_train_to_select_best_model(merged_file, ratio_train_file):
    final_scores = {
        0: {"type":"obj_count", "scores":[0.00, 0.00, 0.00, 0.00, 0.00, 0.00], "count":0, "avg_score":[0.00, 0.00, 0.00, 0.00, 0.00, 0.00], "best_model":""},
        1: {"type":"odd_even", "scores":[0.00, 0.00, 0.00, 0.00, 0.00, 0.00], "count":0, "avg_score":[0.00, 0.00, 0.00, 0.00, 0.00, 0.00], "best_model":""},
        2: {"type":"size", "scores":[0.00, 0.00, 0.00, 0.00, 0.00, 0.00], "count":0, "avg_score":[0.00, 0.00, 0.00, 0.00, 0.00, 0.00], "best_model":""},
        3: {"type":"front_back", "scores":[0.00, 0.00, 0.00, 0.00, 0.00, 0.00], "count":0, "avg_score":[0.00, 0.00, 0.00, 0.00, 0.00, 0.00], "best_model":""},
        4: {"type":"left_right", "scores":[0.00, 0.00, 0.00, 0.00, 0.00, 0.00], "count":0, "avg_score":[0.00, 0.00, 0.00, 0.00, 0.00, 0.00], "best_model":""},
        5: {"type":"nearest", "scores":[0.00, 0.00, 0.00, 0.00, 0.00, 0.00], "count":0, "avg_score":[0.00, 0.00, 0.00, 0.00, 0.00, 0.00], "best_model":""}
    }

    data = read_json_file(merged_file)
    train_data = read_txt_file(ratio_train_file)
    print(f"train_data length:", len(train_data))
    train_ids = set()
    for line in train_data:
        id = line.split('.')[0].split('/')[-1].lstrip('0').strip() # 处理得到数据id
        train_ids.add(int(id))
    models = MODELS.keys()
    for item in data:
        id = item['id']
        if id in train_ids: # 只对训练集进行评估
            category = item['type_info']
            final_scores[category]["count"] += 1
            for model in models:
                pred = item['models'][model]
                target = item['answer']
                
                if category in [0,2]:
                    try:
                        pred_float = float(pred)
                    except ValueError:
                        pred_float = 0.00
                    mra = cal_mra_cly(pred_float, float(target))
                    final_scores[category]["scores"][MODELS[model]] += mra
                elif category in [1, 3, 4, 5]:
                    if pred == target:
                        final_scores[category]["scores"][MODELS[model]] += 1
                else:
                    print("Error: category not found")
                    print(item)
                    exit()
    for category in final_scores.keys():
        for i in range(len(final_scores[category]["scores"])):
            if final_scores[category]["scores"][i] != 0:
                final_scores[category]["avg_score"][i] = final_scores[category]["scores"][i] / final_scores[category]["count"]
            else:
                final_scores[category]["avg_score"][i] = 0.00
    
    for i in range(6):
        best_idx = final_scores[i]["avg_score"].index(max(final_scores[i]["avg_score"]))
        final_scores[i]["best_model"] = get_key_from_value(MODELS, best_idx)
    
    print(final_scores)
    return final_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_file", type=str, default="/sda/kongming/3d-cake/script/MoLP/backgroundata/modelresults/merged_v4.json")
    parser.add_argument("--ratio_val_file", type=str, default="/sda/kongming/3d-cake/script/MoLP/training/split/v4/ratio_0.1/val_set_0.1.txt")
    parser.add_argument("--ratio_train_file", type=str, default="/sda/kongming/3d-cake/script/MoLP/training/split/v4/ratio_0.1/train_set_0.1.txt")
    parser.add_argument("--output_file", type=str, default="/sda/kongming/3d-cake/script/MoLP/evaluate/v4/ratio_0.1/evaluate_on_train_label_results.json")
    args = parser.parse_args()

    best_list = eval_on_train_to_select_best_model(args.merged_file, args.ratio_train_file)
    evaluate_based_on_labels(args.merged_file, args.ratio_val_file, args.output_file, best_list)