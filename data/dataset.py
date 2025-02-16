import torch
from torch.utils.data import Dataset
from pathlib import Path

class AnswerDataset(Dataset):
    def __init__(self, emb_dir, label_dir):
        self.emb_dir = Path(emb_dir)
        self.label_dir = Path(label_dir)
        self.length = len(list(self.emb_dir.glob("*.pt")))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        embedding = torch.load(self.emb_dir/f"{idx}.pt")
        label = torch.load(self.label_dir/f"{idx}.pt")
        return embedding, label