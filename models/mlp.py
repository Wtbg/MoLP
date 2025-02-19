import torch.nn as nn

class ConfidenceMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        input_dim = config.model.embedding_dim
        hidden_dims = config.model.hidden_dims
        output_dim = config.model.num_collections
        dropout_rate = config.model.dropout_rate  # 从配置中获取 Dropout 率
        
        # 动态构建隐藏层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # 添加 Dropout 层
            input_dim = h_dim
        
        # 最终输出层
        layers.append(nn.Linear(input_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)