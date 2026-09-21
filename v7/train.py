"""
LeWM 训练入口
"""
import hydra
from omegaconf import DictConfig
from data import DataStrategyRegistry
from core.trainer import Trainer


@hydra.main(version_base=None, config_path="./configs/train", config_name="lewm")
def main(cfg: DictConfig):
    # 1. 自动检测数据格式
    data_path = cfg.data.dataset.name
    strategy = DataStrategyRegistry.detect(data_path)
    print(f"[Train] Using: {strategy.get_name()} | Channels: {strategy.get_input_channels()}")
    
    # 2. 创建 Trainer 并运行
    trainer = Trainer(cfg, strategy)
    trainer.run()


if __name__ == "__main__":
    main()