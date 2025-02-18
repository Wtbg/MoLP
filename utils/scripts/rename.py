import os
import re

def rename_files():
    # 目录路径 (请根据实际情况调整)
    embedding_dir = 'data/processed/labels'
    
    # 匹配类似 "123.pt" 的文件名
    pattern = re.compile(r'^(\d+)\.pt$')

    for filename in os.listdir(embedding_dir):
        match = pattern.match(filename)
        if match:
            idx = match.group(1)
            new_idx = idx.zfill(8)  # 补齐到8位
            new_filename = f"{new_idx}.pt"
            old_path = os.path.join(embedding_dir, filename)
            new_path = os.path.join(embedding_dir, new_filename)
            if filename != new_filename:
                print(f"Renaming {filename} -> {new_filename}")
                os.rename(old_path, new_path)

if __name__ == "__main__":
    rename_files()