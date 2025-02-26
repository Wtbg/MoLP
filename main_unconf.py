"""
主训练脚本 - 包含完整的训练验证循环、检查点保存和指标记录
"""
import argparse
import pdb
import torch
from torch.utils.data import DataLoader, random_split
from utils.config import load_config
from utils.logger import setup_logger
from data.dataset import AnswerDataset
from models import ConfidenceMLP
from training.trainer import Trainer
from training.evaluator import Evaluator
from training.checkpoint import save_best_model
import os
from datetime import datetime
from pathlib import Path

def main(args):
    # pdb.set_trace()
    # ------------------ 初始化阶段 ------------------ 
    # 加载配置
    config = load_config("configs/default.yaml")
    config.paths.train_data_dir = args.train_data_dir
    config.paths.train_label_dir = args.train_label_dir
    config.paths.val_data_dir = args.val_data_dir
    config.paths.val_label_dir = args.val_label_dir
    config.paths.log_dir = args.log_dir
    config.paths.checkpoint_dir = args.checkpoint_dir
    logger = setup_logger(config.paths.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    
    # ------------------ 数据准备 ------------------
    logger.info("Preparing datasets...")
    
    # 使用完整数据集作为训练集和验证集
    train_dataset = AnswerDataset(
        emb_dir=config.paths.train_data_dir,
        label_dir=config.paths.train_label_dir
    )
    val_dataset = AnswerDataset(
        emb_dir=config.paths.val_data_dir,
        label_dir=config.paths.val_label_dir
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=False
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
            # pdb.set_trace()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--train_label_dir", type=str, default=None)
    parser.add_argument("--val_data_dir", type=str, default=None)
    parser.add_argument("--val_label_dir", type=str, default=None)
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # check if the directory exists
    if not Path(args.train_data_dir).exists():
        raise FileNotFoundError(f"train_data_dir does not exist: {args.train_data_dir}")
    if not Path(args.train_label_dir).exists():
        raise FileNotFoundError(f"train_label_dir does not exist: {args.train_label_dir}")
    if not Path(args.val_data_dir).exists():
        raise FileNotFoundError(f"val_data_dir does not exist: {args.val_data_dir}")
    if not Path(args.val_label_dir).exists():
        raise FileNotFoundError(f"val_label_dir does not exist: {args.val_label_dir}")
    main(args)