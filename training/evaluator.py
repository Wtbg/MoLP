import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class Evaluator:
    def __init__(self, model, val_loader, device, logger):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for embeddings, labels in tqdm(self.val_loader, desc=f"Validating Epoch {epoch}"):
                embeddings = embeddings.to(self.device)
                labels = labels.float().to(self.device)
                
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                
                # 记录数据
                total_loss += loss.item()
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(
            (np.array(all_labels) > 0.5).astype(int),
            (np.array(all_preds) > 0.5).astype(int)
        )
        
        # 记录日志
        self.logger.info(f"Validation Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
        return {'loss': avg_loss, 'accuracy': acc}