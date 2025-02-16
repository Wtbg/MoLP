import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

class FrozenEmbedding(nn.Module):
    """冻结的Qwen文本嵌入生成器"""
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        self.model.eval()

    def forward(self, texts):
        """输入文本列表，返回嵌入向量"""
        with torch.no_grad():
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)  # 平均池化