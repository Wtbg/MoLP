from pathlib import Path
import pdb
import torch
import logging


def check_embeddings(embedding_dir):
    """检查嵌入文件"""
    emb_dir = Path(embedding_dir)
    emb_files = sorted(list(emb_dir.glob("*.pt")))
    for emb_file in emb_files:
        emb = torch.load(emb_file)
        # pdb.set_trace()
        assert emb.dim() == 1, f"维度错误: {emb_file}"
        assert emb.size(0) == 1536, f"维度错误: {emb_file}"
        
def check_labels(label_dir):
    """检查标签文件"""
    label_dir = Path(label_dir)
    label_files = sorted(list(label_dir.glob("*.pt")))
    for label_file in label_files:
        label = torch.load(label_file)
        try:
            assert label.dim() == 1, f"维度错误: {label_file}. Expected 1D, got {label.dim()}."
            assert label.size(0) == 6, f"维度错误: {label_file}. Expected size 6, got {label.size(0)}."
        except AssertionError as err:
            print('Error in file:', label_file)
            logging.error(err)
        
if __name__ == "__main__":
    check_embeddings("data/v4/processed/embeddings")
    check_labels("data/v4/processed/labels")