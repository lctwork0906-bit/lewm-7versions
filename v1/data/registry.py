from typing import Dict, Type, Optional, List
from .base import DataFormatStrategy


class DataStrategyRegistry:
    _strategies: Dict[str, Type[DataFormatStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[DataFormatStrategy]) -> None:
        if not issubclass(strategy_class, DataFormatStrategy):
            raise TypeError(f"{strategy_class} must inherit from DataFormatStrategy")
        cls._strategies[name] = strategy_class
        print(f"[Registry] Registered: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Type[DataFormatStrategy]]:
        return cls._strategies.get(name)

    @classmethod
    def detect(cls, path: str) -> DataFormatStrategy:
        for name, strategy_class in cls._strategies.items():
            strategy = strategy_class()
            try:
                if strategy.detect(path):
                    print(f"[Registry] Auto-detected: {name}")
                    return strategy
            except Exception:
                continue
        raise ValueError(f"No strategy found for: {path}")

    @classmethod
    def list_strategies(cls) -> List[str]:
        return list(cls._strategies.keys())