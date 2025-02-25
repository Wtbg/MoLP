import torch

num_pt_file = "/sda/kongming/3d-cake/script/MoLP/data/v4/processed/labels/00001118.pt"
mcq_pt_file = "/sda/kongming/3d-cake/script/MoLP/data/v4/processed/labels/00000001.pt"

embed_num = torch.load(num_pt_file)
print(embed_num)

embed_mcq = torch.load(mcq_pt_file)
print(embed_mcq)