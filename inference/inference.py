import pdb
import torch
import json
from pathlib import Path

from tqdm import tqdm
from utils.config import load_config
from data.dataset import AnswerDataset
from models import ConfidenceMLP
from transformers import AutoTokenizer, AutoModel
from utils.logger import setup_logger

def load_model(checkpoint_path, config, device):
    model = ConfidenceMLP(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def preprocess_question(question, tokenizer, llm_model, device):
    # 假设有一个 tokenizer 来处理问题文本
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)
    embeddings = llm_model(**inputs).last_hidden_state.mean(dim=1).squeeze().to(device)
    return embeddings

def predict(model, question, tokenizer, llm_model, device):
    embeddings = preprocess_question(question, tokenizer, llm_model, device)
    with torch.no_grad():
        outputs = model(embeddings)
        probs = torch.sigmoid(outputs).cpu().numpy()
    return probs

def load_questions(questions_path):
    with open(questions_path, "r") as f:
        questions = json.load(f)
    return questions

def main():
    config = load_config("configs/default.yaml")
    logger = setup_logger(config.paths.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    checkpoint_path = Path(config.paths.checkpoint_dir) / "best_model.pt"
    model = load_model(checkpoint_path, config, device)
    
    questions = load_questions("raw_data/questions/questions_v4.json")
    
    tokenizer = tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-1.5B-Instruct")
    llm_model = AutoModel.from_pretrained("Qwen2.5-1.5B-Instruct").to(device)
    llm_model.eval()
    
    results = []
    for q in tqdm(questions, desc="Predicting"):
        probs = predict(model, q["question"], tokenizer, llm_model, device)
        result_item = {
            "id": q["id"],
            "question": q["question"],
            "label": probs.tolist()
        }
        results.append(result_item)
    
    # 保存结果
    with open("inference/results_v4.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()