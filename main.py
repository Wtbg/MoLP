"""
主训练脚本 - 包含完整的训练验证循环、检查点保存和指标记录
"""
import torch
from torch.utils.data import DataLoader, random_split
from utils.config import load_config
from utils.logger import setup_logger
from data.dataset import AnswerDataset
from models import ConfidenceMLP
from training.trainer import Trainer
from training.evaluator import Evaluator
from training.checkpoint import save_best_model

def main():
    # ------------------ 初始化阶段 ------------------ 
    # 加载配置
    config = load_config("configs/default.yaml")
    logger = setup_logger(config.paths.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # ------------------ 数据准备 ------------------
    logger.info("Preparing datasets...")
    # 加载完整数据集
    full_dataset = AnswerDataset(
        emb_dir=config.paths.train_emb_dir,
        label_dir=config.paths.train_label_dir
    )
    
    # 划分训练集和验证集
    val_size = int(config.data.val_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    # ------------------ 模型初始化 ------------------
    logger.info("Initializing model...")
    model = ConfidenceMLP(config)
    model.to(device)
    logger.info(f"Model architecture:\n{model}")
    
    # ------------------ 训练准备 ------------------
    logger.info("Initializing training components...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        device=device,
        logger=logger
    )
    evaluator = Evaluator(
        model=model,
        val_loader=val_loader,
        device=device,
        logger=logger
    )
    
    # 跟踪最佳指标
    best_metric = float('inf') if config.train.metric_mode == 'min' else -float('inf')
    
    # ------------------ 训练循环 ------------------
    logger.info(f"Starting training for {config.train.epochs} epochs...")
    for epoch in range(1, config.train.epochs + 1):
        # 训练阶段
        train_loss = trainer.train_epoch(epoch)
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")
        
        # 验证阶段
        if epoch % config.train.eval_interval == 0:
            val_metrics = evaluator.evaluate(epoch)
            
            # 保存最佳模型
            current_metric = val_metrics[config.train.metric_name]
            if (config.train.metric_mode == 'min' and current_metric < best_metric) or \
               (config.train.metric_mode == 'max' and current_metric > best_metric):
                best_metric = current_metric
                save_best_model(
                    model=model,
                    epoch=epoch,
                    metric=best_metric,
                    save_path=config.paths.checkpoint_dir
                )
                logger.info(f"New best {config.train.metric_name} achieved: {best_metric:.4f}")
        
        # 定期保存检查点
        if epoch % config.train.save_interval == 0:
            trainer.save_checkpoint(epoch, train_loss)
            logger.info(f"Saved checkpoint at epoch {epoch}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()