"""
HDF5 + RGBD (4通道: RGB + Depth) 数据策略
基于 RGB 策略扩展，支持深度通道
"""
import hdf5plugin
import h5py
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from typing import Any, Optional
from ..base import DataFormatStrategy


class HDF5RGBDDataset(Dataset):
    def __init__(self, h5_path: str, keys_to_load: list, num_steps: int = 1, transform: Optional[Any] = None):
        self.h5_path = h5_path
        self.keys_to_load = keys_to_load
        self.num_steps = num_steps
        self.transform = transform
        self._load_metadata()

    def _load_metadata(self):
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['pixels'].shape[0] - self.num_steps + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            data = {}
            for key in self.keys_to_load:
                if key in f:
                    if key == "pixels":
                        # RGBD 是 4 通道 (RGB + Depth)
                        if self.num_steps == 1:
                            data[key] = torch.from_numpy(f[key][idx]).permute(2, 0, 1)
                        else:
                            frames = [torch.from_numpy(f[key][idx + i]).permute(2, 0, 1) for i in range(self.num_steps)]
                            data[key] = torch.stack(frames, dim=0)
                    elif key == "action":
                        if self.num_steps == 1:
                            data[key] = torch.from_numpy(f[key][idx])
                        else:
                            frames = [torch.from_numpy(f[key][idx + i]) for i in range(self.num_steps)]
                            data[key] = torch.stack(frames, dim=0)
                    else:
                        data[key] = torch.from_numpy(f[key][idx])
            if self.transform:
                data = self.transform(data)
            return data


class HDF5RGBDStrategy(DataFormatStrategy):
    def detect(self, path: str) -> bool:
        try:
            with h5py.File(path, 'r') as f:
                if 'pixels' not in f:
                    return False
                pixels = f['pixels']
                # 检测是否为 4 通道 (RGBD)
                if len(pixels.shape) == 4:
                    if pixels.shape[-1] == 4 or pixels.shape[1] == 4:
                        return True
                # 或者检测是否有 depth 字段
                if 'depth' in f or 'weight' in f:
                    return True
            return False
        except Exception:
            return False

    def load(self, path: str, **kwargs) -> Dataset:
        keys_to_load = kwargs.get('keys_to_load', ['pixels', 'action'])
        transform = kwargs.get('transform', None)
        num_steps = kwargs.get('num_steps', 1)
        return HDF5RGBDDataset(path, keys_to_load, num_steps, transform)

    def get_input_channels(self) -> int:
        return 4

    def get_transform(self, img_size: int):
        # RGBD: RGB 用 ImageNet，Depth 归一化到 [0,1]
        # 4 通道的 mean/std
        mean = [0.485, 0.456, 0.406, 0.5]
        std = [0.229, 0.224, 0.225, 0.5]

        return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=mean, std=std),
            transforms.Resize(size=img_size),
        ])

    def get_column_names(self) -> list:
        return ['pixels', 'action', 'depth', 'proprio', 'state']

    def supports_depth(self) -> bool:
        return True

    def get_depth_column(self) -> str:
        return 'depth'

    def get_name(self) -> str:
        return "HDF5-RGBD"
