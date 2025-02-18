import argparse
import json
from pathlib import Path
import pdb
import numpy as np
import torch
from tqdm import tqdm

def calculate_mra_array(y_true, y_pred):
    """
    计算平均相对准确率 (MRA)。

    参数：
    y_true: 真实值数组。
    y_pred: 预测值数组。

    返回：
    MRA 值。
    """

    if len(y_true) != len(y_pred):
        raise ValueError("真实值和预测值的长度必须相同。")

    C = np.arange(0.5, 1.0, 0.05)  # 创建阈值集合
    mra = 0

    for theta in C:
        for i in range(len(y_true)):
            if y_true[i] != 0: # 避免除以0的错误
                relative_error = abs(y_pred[i] - y_true[i]) / y_true[i]
                if relative_error < (1 - theta):
                    mra += 1
            else:
                if abs(y_pred[i] - y_true[i]) < (1 - theta):
                    mra +=1
                
    mra /= (len(C) * len(y_true))
    return mra

def calculate_mra_single(y_true, y_pred):
    """
    计算单个阈值下的平均相对准确率 (MRA)。

    参数：
    y_true: 真实值
    y_pred: 预测值

    返回：
    MRA 值。
    """
    C = np.arange(0.5, 1.0, 0.05)  # 创建阈值集合
    mra = 0
    
    for theta in C:
        if y_true != 0:
            relative_error = abs(y_pred - y_true) / y_true
            if relative_error < (1 - theta):
                mra += 1
        else:
            if abs(y_pred - y_true) < (1 - theta):
                mra += 1
    
    mra /= len(C)
    return mra

def calculate_loose_mra_single(y_true, y_pred):
    """
    计算单个阈值下的平均相对准确率 (MRA)。

    参数：
    y_true: 真实值
    y_pred: 预测值

    返回：
    MRA 值。
    """
    C = np.arange(0.0, 1.0, 0.05)  # 创建阈值集合
    mra = 0
    
    for theta in C:
        if y_true != 0:
            relative_error = abs(y_pred - y_true) / y_true
            if relative_error < (1 - theta):
                mra += 1
        else:
            if abs(y_pred - y_true) < (1 - theta):
                mra += 1
    
    mra /= len(C)
    return mra

def generate_label(result_file, label_dir):
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'r') as f:
        data = json.load(f)
    for item in tqdm(data, desc="Generating labels"):
        if item['type_info'] not in [0, 2]:
            answer = item['answer']
            labels = []
            for model, pred in item['models'].items():
                if pred == answer:
                    labels.append(1)
                else:
                    labels.append(0)
        elif item['type_info'] in [0, 2]:
            # pdb.set_trace()
            answer = item['answer']
            labels = []
            for model, pred in item['models'].items():
                try:
                    answer = float(answer)
                    pred = float(pred)
                    mra = calculate_loose_mra_single(answer, pred)
                except ValueError:
                    mra = 0
                labels.append(mra)
            # 使用sigmoid归一化
            labels = torch.sigmoid(torch.tensor(labels)).tolist()
            if all(label == 0 for label in labels):
                labels = [0.0 for _ in labels]
            else:
                labels = [label / max(labels) for label in labels]
        else:
            raise ValueError("Invalid question type.")
        item_id = str(item['id']).zfill(8)
        torch.save(torch.tensor(labels), label_dir/f"{item_id}.pt")
        print(labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default="backgroundata/modelresults/merged.json")
    parser.add_argument("--label_dir", type=str, default="data/processed/labels")
    args = parser.parse_args()
    generate_label(args.result_file, args.label_dir)
    print(f"Labels saved to {args.label_dir}")