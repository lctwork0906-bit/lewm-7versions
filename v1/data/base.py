from abc import ABC, abstractmethod
from typing import Any, Optional
from torch.utils.data import Dataset


class DataFormatStrategy(ABC):
    """数据格式策略抽象基类"""

    @abstractmethod
    def detect(self, path: str) -> bool:
        """检测是否支持此格式"""
        pass

    @abstractmethod
    def load(self, path: str, **kwargs) -> Dataset:
        """加载数据集"""
        pass

    @abstractmethod
    def get_input_channels(self) -> int:
        """返回输入通道数（3 或 4）"""
        pass

    @abstractmethod
    def get_transform(self, img_size: int) -> Any:
        """返回数据预处理 transform"""
        pass

    @abstractmethod
    def get_column_names(self) -> list:
        """返回数据集包含的字段名列表"""
        pass

    def supports_depth(self) -> bool:
        return False

    def get_depth_column(self) -> Optional[str]:
        return None

    def get_name(self) -> str:
        return self.__class__.__name__