import argparse
import json
from pathlib import Path
import pdb
from utils.scripts.generate_label import calculate_mra_single, calculate_mra_array, mean_relative_accuracy

def cal_mra_cly(pred, target):
    mra = mean_relative_accuracy(pred, target, .5, .95, .05)
    return mra

def load_merge_results(result_files):
    with open(result_files, "r") as f:
        results = json.load(f)
    return results

def load_inference_results(result_files):
    with open(result_files, "r") as f:
        results = json.load(f)
    return results

def load_id_set(id_set_file):
    with open(id_set_file, "r") as f:
        id_set = f.readlines()
    # data/processed/embeddings/00000001.pt
    # 提取出id, 不含有先导0
    id_set = [id.split("/")[-1].split(".")[0].lstrip("0") for id in id_set]
    id_set = [int(id) for id in id_set]
    return id_set

def answer_on_set(merge_results, inference_results, id_set):
    label_map = {}
    for entry in inference_results:
        label_map[entry["id"]] = entry["label"]
    moe_pred_map = {}
    gt_map = {}
    type_info_map = {}
    for entry in merge_results:
        if entry['type_info'] not in [0, 2]:
            answer_vote_map = {
                "A": 0,
                "B": 0,
                "C": 0,
                "D": 0
            }
            topk = 2
            inference_label = label_map[entry["id"]]
            model_answers = []
            for id, (model, pred) in enumerate(entry["models"].items()):
                model_answers.append(pred)
            topk_idx = sorted(range(len(inference_label)), key=lambda x: inference_label[x], reverse=True)[:topk]
            for idx in topk_idx:
                answer_vote_map[model_answers[idx]] += inference_label[idx]
            moe_pred = max(answer_vote_map, key=answer_vote_map.get)
            # pdb.set_trace()
            if args.if_evaluate_benchmark == True:
                moe_pred = dict(entry["models"]).get(args.evaluate_benchmark_name)
            entry["moe_pred"] = moe_pred
            moe_pred_map[entry["id"]] = moe_pred
            gt_map[entry["id"]] = entry["answer"]
            type_info_map[entry["id"]] = entry["type_info"]
        else:
            model_answers = []
            inference_label = label_map[entry["id"]]
            for id, (model, pred) in enumerate(entry["models"].items()):
                # catch float() exception
                try:
                    model_answers.append(float(pred))
                except:
                    model_answers.append(0.0)
            # 选择置信权重最大的模型的答案作为最终答案
            moe_pred = model_answers[inference_label.index(max(inference_label))]
            # pdb.set_trace()
            if args.if_evaluate_benchmark == True:
                moe_pred = dict(entry["models"]).get(args.evaluate_benchmark_name)
                try:
                    moe_pred = float(moe_pred)
                except:
                    if(moe_pred == "TWO"):
                        moe_pred = 2.0
                    else:
                        moe_pred = 0.0
            moe_pred_map[entry["id"]] = moe_pred
            gt_map[entry["id"]] = float(entry["answer"])
            type_info_map[entry["id"]] = entry["type_info"]
    type1_cnt = 0
    type3_cnt = 0
    type4_cnt = 0
    type5_cnt = 0
    type1_correct_cnt = 0
    type3_correct_cnt = 0
    type4_correct_cnt = 0
    type5_correct_cnt = 0
    type0_cnt = 0
    type2_cnt = 0
    type0_mra_sum = 0
    type2_mra_sum = 0
    for id in id_set:
        if id in gt_map:
            id = int(id)
            # 检查id在三个map中是否都存在
            assert id in moe_pred_map
            assert id in gt_map
            assert id in type_info_map
            if type_info_map[id] == 0:
                type0_cnt += 1
                # type0_mra_sum += calculate_mra_single(moe_pred_map[id], gt_map[id])
                type0_mra_sum += cal_mra_cly(moe_pred_map[id], gt_map[id])
            elif type_info_map[id] == 1:
                type1_cnt += 1
                if moe_pred_map[id] == gt_map[id]:
                    type1_correct_cnt += 1
            elif type_info_map[id] == 2:
                type2_cnt += 1
                # type2_mra_sum += calculate_mra_single(moe_pred_map[id], gt_map[id])
                type2_mra_sum += cal_mra_cly(moe_pred_map[id], gt_map[id])  
            elif type_info_map[id] == 3:
                type3_cnt += 1
                if moe_pred_map[id] == gt_map[id]:
                    type3_correct_cnt += 1
            elif type_info_map[id] == 4:
                type4_cnt += 1
                if moe_pred_map[id] == gt_map[id]:
                    type4_correct_cnt += 1
            elif type_info_map[id] == 5:
                type5_cnt += 1
                if moe_pred_map[id] == gt_map[id]:
                    type5_correct_cnt += 1
    type0_mra = type0_mra_sum / type0_cnt
    type2_mra = type2_mra_sum / type2_cnt
    type1_accuracy = type1_correct_cnt / type1_cnt
    type3_accuracy = type3_correct_cnt / type3_cnt
    type4_accuracy = type4_correct_cnt / type4_cnt
    type5_accuracy = type5_correct_cnt / type5_cnt
    # 根据type_info_map和label_map统计各个type_info对每个模型答案的置信度的均值和方差
    type_info_average = {}
    type_info_variance = {}
    for id, label in label_map.items():
        type_info = type_info_map[id]
        if type_info not in type_info_average:
            type_info_average[type_info] = [0] * len(label)
            type_info_variance[type_info] = [0] * len(label)
        type_info_average[type_info] = [a + b for a, b in zip(type_info_average[type_info], label)]
    return {
        "type0_mra": type0_mra,
        "type2_mra": type2_mra,
        "type1_accuracy": type1_accuracy,
        "type3_accuracy": type3_accuracy,
        "type4_accuracy": type4_accuracy,
        "type5_accuracy": type5_accuracy
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge_results", type=str, default="backgroundata/modelresults/merged_v4.json")
    parser.add_argument("--inference_results", type=str, default="inference/results_v4.json")
    parser.add_argument("--id_set", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--evaluate_benchmark_name", type=str, default=None)
    parser.add_argument("--if_evaluate_benchmark", type=bool, default=False)
    args = parser.parse_args()
    id_set_file = args.id_set
    id_set = load_id_set(id_set_file)
    merge_results = load_merge_results(args.merge_results)
    inference_results = load_inference_results(args.inference_results)
    result = answer_on_set(merge_results, inference_results, id_set)
    print(result)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.if_evaluate_benchmark == True:
        result_file = f"{args.output_dir}/evaluate_results_{args.evaluate_benchmark_name}.json"
        with open(result_file, "w") as f:
            json.dump(result, f)
    else:
        result_file = f"{args.output_dir}/evaluate_results.json"
        with open(result_file, "w") as f:
            json.dump(result, f)
    