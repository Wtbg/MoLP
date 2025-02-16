import yaml
from argparse import Namespace
from pathlib import Path

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 递归转换为Namespace对象
        self.config = self._dict_to_namespace(config_dict)
        
        # 处理路径
        self._process_paths()
    
    def _dict_to_namespace(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = self._dict_to_namespace(v)
        return Namespace(**d)
    
    def _process_paths(self):
        # 确保所有路径存在
        paths = self.config.paths.__dict__
        for name, path in paths.items():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def __getattr__(self, name):
        return getattr(self.config, name)

def load_config(config_path):
    return Config(config_path).config