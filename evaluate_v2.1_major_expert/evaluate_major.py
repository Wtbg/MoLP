import argparse
import json
import pdb
import sys
import numpy as np
import os
# sys.path.append("..")
sys.path.append("/sda/kongming/3d-cake/script/MoLP")
from utils.scripts.generate_label import calculate_mra_single, calculate_mra_array


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


def calculate_results(label_map, moe_pred_map, gt_map, type_info_map):
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
                type0_mra_sum += calculate_mra_single(moe_pred_map[id], gt_map[id])
            elif type_info_map[id] == 1:
                type1_cnt += 1
                if moe_pred_map[id] == gt_map[id]:
                    type1_correct_cnt += 1
            elif type_info_map[id] == 2:
                type2_cnt += 1
                type2_mra_sum += calculate_mra_single(moe_pred_map[id], gt_map[id])
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
    # # 根据type_info_map和label_map统计各个type_info对每个模型答案的置信度的均值和方差
    # type_info_average = {}
    # type_info_variance = {}
    # for id, label in label_map.items():
    #     type_info = type_info_map[id]
    #     if type_info not in type_info_average:
    #         type_info_average[type_info] = [0] * len(label)
    #         type_info_variance[type_info] = [0] * len(label)
    #     type_info_average[type_info] = [a + b for a, b in zip(type_info_average[type_info], label)]
    return {
        "type0_mra": type0_mra,
        "type2_mra": type2_mra,
        "type1_accuracy": type1_accuracy,
        "type3_accuracy": type3_accuracy,
        "type4_accuracy": type4_accuracy,
        "type5_accuracy": type5_accuracy
    }




def answer_on_set_vanilla_major(merge_results, inference_results, id_set, major_expert, major_weight, top_k):

    num_label_map = {} # 获取模型预测结果logits
    mcq_label_map = {} # 获取模型预测结果logits

    major_idx = list(merge_results[0]["models"].keys()).index(major_expert) # 获取major_expert在labels中对应的下标

    # 删去major expert的logit, 归一化后, 结果存入mcq_label_map
    for entry in inference_results:
        # for num
        num_label_map[entry["id"]] = entry["label"]
        # for mcq
        temp = entry["label"][:] # 用切片, 避免对原列表产生影响
        del temp[major_idx]
        labels = np.array(temp)
        softmax_values = np.exp(labels) / np.sum(np.exp(labels))
        temp = softmax_values.tolist()
        mcq_label_map[entry["id"]] = temp

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
            topk = top_k # ====top-k在此处设置====
            inference_label = mcq_label_map[entry["id"]] # 当前id的预测logits(此处已删去了major_expert)
            model_answers = []
            for id, (model, pred) in enumerate(entry["models"].items()):
                if model == major_expert: # 将major_expert的答案作为 major_answer
                    major_answer = pred
                else: # 删除major_expert的选项
                    model_answers.append(pred)
            # moe_pred = model_answers[inference_label.index(max(inference_label))]
            # 找到pred中topk的idx们
            topk_idx = sorted(range(len(inference_label)), key=lambda x: inference_label[x], reverse=True)[:topk]
            for idx in topk_idx: # 逐个叠加topk的答案结果logits
                answer_vote_map[model_answers[idx]] += inference_label[idx]
            answer_vote_map[major_answer] += major_weight # 将major_expert的权重加到对应的答案上
            moe_pred = max(answer_vote_map, key=answer_vote_map.get) # 获得最终的moe答案
            
            entry["moe_pred"] = moe_pred
            moe_pred_map[entry["id"]] = moe_pred
            gt_map[entry["id"]] = entry["answer"]
            type_info_map[entry["id"]] = entry["type_info"]
        else:
            model_answers = []
            inference_label = num_label_map[entry["id"]]
            for id, (model, pred) in enumerate(entry["models"].items()):
                # catch float() exception
                try:
                    model_answers.append(float(pred))
                except:
                    model_answers.append(0.0)
            # 选择置信权重最大的模型的答案作为最终答案
            moe_pred = model_answers[inference_label.index(max(inference_label))]
            moe_pred_map[entry["id"]] = moe_pred
            gt_map[entry["id"]] = float(entry["answer"])
            type_info_map[entry["id"]] = entry["type_info"]
    results = calculate_results(num_label_map, moe_pred_map, gt_map, type_info_map)
    return results


def answer_on_set_topk_minus_lastk_major(merge_results, inference_results, id_set, major_expert, major_weight, top_k, last_k):
    num_label_map = {} # 获取模型预测结果logits
    mcq_label_map = {} # 获取模型预测结果logits

    major_idx = list(merge_results[0]["models"].keys()).index(major_expert) # 获取major_expert在labels中对应的下标

    # 删去major expert的logit, 归一化后, 结果存入mcq_label_map
    for entry in inference_results:
        # for num
        num_label_map[entry["id"]] = entry["label"]
        # for mcq
        temp = entry["label"][:] # 用切片, 避免对原列表产生影响
        del temp[major_idx]
        labels = np.array(temp)
        softmax_values = np.exp(labels) / np.sum(np.exp(labels))
        temp = softmax_values.tolist()
        mcq_label_map[entry["id"]] = temp
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
            topk = top_k # ====top-k在此处设置====
            lastk = last_k # ====last-k在此处设置====
            inference_label = mcq_label_map[entry["id"]] # 当前id的预测logits
            model_answers = []
            for id, (model, pred) in enumerate(entry["models"].items()):
                if model == major_expert: # 将major_expert的答案作为 major_answer
                    major_answer = pred
                else: # 删除major_expert的选项
                    model_answers.append(pred)
            # moe_pred = model_answers[inference_label.index(max(inference_label))]
            # 找到pred中topk的idx们
            topk_idx = sorted(range(len(inference_label)), key=lambda x: inference_label[x], reverse=True)[:topk]
            last_idx = sorted(range(len(inference_label)), key=lambda x: inference_label[x], reverse=True)[-lastk:]
            for idx in topk_idx:
                answer_vote_map[model_answers[idx]] += inference_label[idx] # 逐个加上top-k的选项logits
            answer_vote_map[major_answer] += major_weight # 将major_expert的权重加到对应的答案上
            for idx in last_idx:
                answer_vote_map[model_answers[idx]] -= inference_label[idx] # 减去最后一名的选项logits
            moe_pred = max(answer_vote_map, key=answer_vote_map.get)
                
            entry["moe_pred"] = moe_pred
            moe_pred_map[entry["id"]] = moe_pred
            gt_map[entry["id"]] = entry["answer"]
            type_info_map[entry["id"]] = entry["type_info"]
        else:
            model_answers = []
            inference_label = num_label_map[entry["id"]]
            for id, (model, pred) in enumerate(entry["models"].items()):
                # catch float() exception
                try:
                    model_answers.append(float(pred))
                except:
                    model_answers.append(0.0)
            # 选择置信权重最大的模型的答案作为最终答案
            moe_pred = model_answers[inference_label.index(max(inference_label))]
            moe_pred_map[entry["id"]] = moe_pred
            gt_map[entry["id"]] = float(entry["answer"])
            type_info_map[entry["id"]] = entry["type_info"]
    results = calculate_results(num_label_map, moe_pred_map, gt_map, type_info_map)
    return results




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sabench_version", type=str, default="v2.1") #
    parser.add_argument("--major_expert", type=str, default="llava-3d") # "llavanextvideo7b" "longva7b" "chat-scene" "leo" "video_3d_llm" "llava-3d"
    parser.add_argument("--major_weight", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--lastk", type=int, default=1)
    parser.add_argument("--merge_results", type=str, default="backgroundata/modelresults/merged.json")
    parser.add_argument("--inference_results", type=str, default="inference/results.json")
    parser.add_argument("--id_set", type=str, default="training/split/val_set_0.8.txt")
    args = parser.parse_args()
    id_set_file = args.id_set
    id_set = load_id_set(id_set_file)
    merge_results = load_merge_results(args.merge_results)
    inference_results = load_inference_results(args.inference_results)
    # result = answer_on_set(merge_results, inference_results, id_set) # ld
    # eval(f"{result} = answer_on_set")

    if args.lastk <= 0:
        result = answer_on_set_vanilla_major(merge_results, inference_results, id_set, args.major_expert, args.major_weight, args.topk)
        relative_folder = f"evaluate_{args.sabench_version}_major_expert/ratio_0.8_major/{args.major_expert}"
        output_file = f"evaluate_results_top{args.topk}_major_weight{args.major_weight}.json"
        if not os.path.exists(relative_folder):
            # 如果不存在，则创建文件夹
            os.makedirs(relative_folder)
        with open(f"{relative_folder}/{output_file}", "w") as f:
            json.dump(result, f)
            print(f"Successfully generated: {relative_folder}/{output_file}")
    else:
        result = answer_on_set_topk_minus_lastk_major(merge_results, inference_results, id_set, args.major_expert, args.major_weight, args.topk, args.lastk)
        relative_folder = f"evaluate_{args.sabench_version}_major_expert/ratio_0.8_major/{args.major_expert}"
        output_file = f"evaluate_results_top{args.topk}-last{args.lastk}_major_weight{args.major_weight}.json"
        if not os.path.exists(relative_folder):
            # 如果不存在，则创建文件夹
            os.makedirs(relative_folder)
        with open(f"{relative_folder}/{output_file}", "w") as f:
            json.dump(result, f)
            print(f"Successfully generated: {relative_folder}/{output_file}")
