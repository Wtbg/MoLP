import torch
from torch.utils.data import Dataset
from pathlib import Path

class AnswerDataset(Dataset):
    def __init__(self, emb_dir, label_dir):
        self.emb_dir = Path(emb_dir)
        self.label_dir = Path(label_dir)
        self.emb_files = sorted(list(self.emb_dir.glob("*.pt")))
        self.label_files = sorted(list(self.label_dir.glob("*.pt")))
        self.length = len(self.emb_files)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        emb_file = self.emb_files[idx]
        label_file = self.label_files[idx]
        embedding = torch.load(emb_file).float()
        label = torch.load(label_file).float()
        return embedding, label