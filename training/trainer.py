from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

class Trainer:
    def __init__(self, model, train_loader, config, device, logger):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config.paths.log_dir)
        self.logger = logger
        
        # 初始化训练组件
        self.optimizer = Adam(model.parameters(), lr=float(config.train.learning_rate))
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (embeddings, labels) in enumerate(progress_bar):
            embeddings = embeddings.to(self.device)
            labels = labels.float().to(self.device)
            
            # 前向传播
            outputs = self.model(embeddings)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录日志
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            # TensorBoard记录
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), global_step)
        
        return total_loss / len(self.train_loader) 
    
    def save_checkpoint(self, epoch, loss):
        checkpoint_dir = Path(self.config.paths.checkpoint_dir)
        ckpt_path = checkpoint_dir/f"model_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, ckpt_path)