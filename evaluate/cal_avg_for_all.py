import json
import os
import argparse
from pathlib import Path
import pdb



MODEL_DICT = {
    # 'longva7b': 'evaluate_results_longva7b.json',
    # 'llavanextvideo7b': 'evaluate_results_llavanextvideo7b.json',
    # 'llava-3d': 'evaluate_results_llava-3d.json',
    # 'video_3d_llm': 'evaluate_results_video_3d_llm.json',
    # 'chat-scene': 'evaluate_results_chat-scene.json',
    # 'leo': 'evaluate_results_leo.json',
    'ours-top1': 'evaluate_results.json'
}

def read_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def write_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def cal_avg_for_single_file(file_name):
    data = read_json(file_name)
    score = 0.000
    for key in data.keys():
        score += float(data[key])
    avg_score = score / len(data.keys())
    return avg_score

def cal_avg_for_all(result_dir):
    ratio_folders = [p.name for p in Path(result_dir).iterdir() if p.is_dir()] # 获取当前文件夹下所有子文件夹
    for folder in ratio_folders:
        ratio_dir = Path(os.path.join(result_dir, folder))
        split_folders = [p.name for p in Path(ratio_dir).iterdir() if p.is_dir()]
        ratio_results = []
        for split_folder in split_folders:
            split_dir = Path(os.path.join(ratio_dir, split_folder))
            avg_results = {}
            for model in MODEL_DICT.keys():
                file_name = os.path.join(split_dir, MODEL_DICT[model])
                avg_score = cal_avg_for_single_file(file_name)
                avg_results[model] = float(avg_score)
            write_json(avg_results, os.path.join(split_dir, 'avg_results.json'))
            ratio_results.append(avg_results)
        write_json(ratio_results, os.path.join(ratio_dir, 'avg_results_all.json'))
        avg_results_all = {}
        for model in MODEL_DICT.keys():
            avg_results_all[model] = 0.0
        for result in ratio_results:
            for model in MODEL_DICT.keys():
                avg_results_all[model] += result[model]
        for model in MODEL_DICT.keys():
            avg_results_all[model] /= len(ratio_results)
        write_json(avg_results_all, os.path.join(ratio_dir, 'all_avg_results_all.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='/sda/kongming/3d-cake/script/MoLP/evaluate/v4_repeat_3part_topk/top_2')
    args = parser.parse_args()

    cal_avg_for_all(args.result_dir)