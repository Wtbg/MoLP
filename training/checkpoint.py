import torch
from pathlib import Path

def save_best_model(model, epoch, metric, save_path):
    """保存最佳模型检查点"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metric': metric
    }
    torch.save(save_dict, Path(save_path)/"best_model.pt")