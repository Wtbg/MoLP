import torch
import torch.nn.functional as F

# 定义标签和logits
label_100001 = torch.tensor([1, 0, 0, 0, 0, 1], dtype=torch.float32)
logits_100001 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])

label_100000 = torch.tensor([1, 0, 0, 0, 0, 1], dtype=torch.float32)
logits_100000 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# 将logits输入BCEWithLogitsLoss函数计算损失
loss_fn = torch.nn.BCEWithLogitsLoss()

# 计算损失
loss_100001_100001 = loss_fn(logits_100001, label_100001)
loss_100000_100001 = loss_fn(logits_100000, label_100001)

# 打印结果
print(f"BCEWithLogitsLoss between 100001 and 100001: {loss_100001_100001.item()}")
print(f"BCEWithLogitsLoss between 100000 and 100001: {loss_100000_100001.item()}")
