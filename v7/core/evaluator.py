"""
评估器（独立实现）
"""
import torch
from tqdm import tqdm
from utils.logging import setup_logger


class Evaluator:
    def __init__(self, cfg, strategy):
        self.cfg = cfg
        self.strategy = strategy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger()
    
    def run(self):
        self.logger.info("Starting evaluation...")
        # 简化：加载模型，跑评估
        # 具体实现根据你的需求
        self.logger.info("Evaluation complete!")