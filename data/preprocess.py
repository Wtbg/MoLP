import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

def generate_embeddings(texts, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
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
        for idx, text in enumerate(texts):
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