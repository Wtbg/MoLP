import json
import pdb
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import argparse

def generate_embeddings(texts, model_name="/sda/kongming/3d-cake/script/MoE/Qwen/Qwen2.5-1.5B-Instruct"):
    """生成文本嵌入并保存"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载冻结的Qwen模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # 创建输出目录
    emb_dir = Path("data/processed/embeddings")
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    # 批量处理文本
    with torch.no_grad():
        for _, unit in tqdm(enumerate(texts), desc="Generating embeddings"):
            # pdb.set_trace()
            text = unit["question"]
            idx = unit["id"]
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            # 使用平均池化获取文本嵌入
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
            torch.save(embedding, emb_dir/f"{idx}.pt")
            
def load_labels(labels):
    """保存标签数据"""
    label_dir = Path("data/processed/labels")
    label_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, label in enumerate(labels):
        torch.save(torch.tensor(label), label_dir/f"{idx}.pt")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="raw_data/questions/questions.json")
    # parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    questions = args.input_dir
    quesions_units = []
    with open(questions, "r") as f:
        data = json.load(f)
    for item in tqdm(data, desc="Extracting questions"):
        # pdb.set_trace()
        quesions_units.append({
            "id": item['id'],
            "question": item['question'],
        })
    generate_embeddings(quesions_units)
    
