"""
LeWM 评估入口
"""
import hydra
from omegaconf import DictConfig
from data import DataStrategyRegistry
from core.evaluator import Evaluator


@hydra.main(version_base=None, config_path="./configs/eval", config_name="pusht")
def main(cfg: DictConfig):
    # 1. 自动检测数据格式
    data_path = cfg.eval.dataset_name
    strategy = DataStrategyRegistry.detect(data_path)
    print(f"[Eval] Using: {strategy.get_name()}")
    
    # 2. 创建 Evaluator 并运行
    evaluator = Evaluator(cfg, strategy)
    evaluator.run()


if __name__ == "__main__":
    main()