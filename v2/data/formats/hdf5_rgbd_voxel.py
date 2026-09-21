"""
HDF5 RGB-D + 体素策略
加载 RGB-D 数据（HDF5），实时转换为 3D 体素
"""
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from typing import Any, Optional

from ..base import DataFormatStrategy
from ..transforms.voxelize import VoxelizeTransform


class HDF5RGBDVoxelDataset(Dataset):
    """
    HDF5 RGB-D 体素数据集
    
    从 HDF5 读取 RGB 和 Depth，实时转换为 3D 体素
    支持多帧堆叠 (num_steps)
    """
    def __init__(
        self,
        h5_path: str,
        keys_to_load: list,
        num_steps: int = 1,
        transform: Optional[Any] = None,
        rgb_key: str = 'pixels',
        depth_key: str = 'depth',
        voxel_key: str = 'voxel',
    ):
        self.h5_path = h5_path
        self.keys_to_load = keys_to_load
        self.num_steps = num_steps
        self.transform = transform
        self.rgb_key = rgb_key
        self.depth_key = depth_key
        self.voxel_key = voxel_key
        self._load_metadata()
        
        # 创建体素转换器（延迟初始化，避免在 DataLoader 子进程中重复创建）
        self._voxelizer = None
    
    def _get_voxelizer(self):
        """延迟初始化体素转换器"""
        if self._voxelizer is None:
            from core.voxel import RGBDVoxelizer, VoxelSpec
            spec = VoxelSpec()
            self._voxelizer = RGBDVoxelizer(spec)
        return self._voxelizer

    def _load_metadata(self):
        with h5py.File(self.h5_path, 'r') as f:
            # 确定数据长度
            if self.rgb_key in f:
                self.length = f[self.rgb_key].shape[0] - self.num_steps + 1
            elif self.depth_key in f:
                self.length = f[self.depth_key].shape[0] - self.num_steps + 1
            else:
                self.length = f[self.keys_to_load[0]].shape[0] - self.num_steps + 1

    def __len__(self):
        return max(0, self.length)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            data = {}
            
            # 读取 RGB 和 Depth，合并为体素
            if self.voxel_key in self.keys_to_load:
                # 直接从 HDF5 读取预计算的体素
                if self.num_steps == 1:
                    voxel = torch.from_numpy(f[self.voxel_key][idx]).float()
                else:
                    frames = [
                        torch.from_numpy(f[self.voxel_key][idx + i]).float()
                        for i in range(self.num_steps)
                    ]
                    voxel = torch.stack(frames, dim=0)
                data[self.voxel_key] = voxel
            
            elif self.rgb_key in f and self.depth_key in f:
                # 实时 RGB-D → 体素转换
                rgb = f[self.rgb_key][idx]
                depth = f[self.depth_key][idx]
                
                # 确保数据格式正确
                if isinstance(rgb, np.ndarray):
                    rgb = torch.from_numpy(rgb)
                if isinstance(depth, np.ndarray):
                    depth = torch.from_numpy(depth)
                
                # 转换：RGB 和 Depth 可能是 (H, W, C) 或 (C, H, W)
                if rgb.ndim == 3 and rgb.shape[-1] == 3:
                    rgb = rgb.numpy()
                elif rgb.ndim == 3 and rgb.shape[0] == 3:
                    rgb = rgb.permute(1, 2, 0).numpy()
                else:
                    rgb = rgb.numpy()
                
                if depth.ndim == 3 and depth.shape[0] == 1:
                    depth = depth.squeeze(0).numpy()
                else:
                    depth = depth.numpy()
                
                # 体素化
                voxelizer = self._get_voxelizer()
                voxel, _ = voxelizer.build(rgb, depth)
                data[self.voxel_key] = voxel
            
            else:
                raise ValueError(
                    f"Need either '{self.voxel_key}' or ('{self.rgb_key}' + '{self.depth_key}') in HDF5"
                )
            
            # 读取其他字段 (action, proprio, state 等)
            for key in self.keys_to_load:
                if key in [self.voxel_key, self.rgb_key, self.depth_key]:
                    continue
                if key in f:
                    if self.num_steps == 1:
                        data[key] = torch.from_numpy(f[key][idx])
                    else:
                        frames = [
                            torch.from_numpy(f[key][idx + i])
                            for i in range(self.num_steps)
                        ]
                        data[key] = torch.stack(frames, dim=0)
            
            # 应用 transform
            if self.transform:
                data = self.transform(data)
            return data


class HDF5RGBDVoxelStrategy(DataFormatStrategy):
    """
    HDF5 RGB-D → 体素策略
    
    检测条件：
    1. 有 'pixels' + 'depth' 字段
    2. 或有 'voxel' 字段（预计算体素）
    3. 或 'pixels' 是 4 通道 (RGBD)
    """
    
    def __init__(self, rgb_key: str = 'pixels', depth_key: str = 'depth', voxel_key: str = 'voxel'):
        self.rgb_key = rgb_key
        self.depth_key = depth_key
        self.voxel_key = voxel_key

    def detect(self, path: str) -> bool:
        """检测是否为 RGB-D 体素格式"""
        try:
            with h5py.File(path, 'r') as f:
                # 方式1：有预计算的体素
                if self.voxel_key in f:
                    return True
                
                # 方式2：有 RGB + Depth
                if self.rgb_key in f and (self.depth_key in f or 'weight' in f):
                    return True
                
                # 方式3：pixels 是 4 通道 (RGBD)
                if self.rgb_key in f:
                    pixels = f[self.rgb_key]
                    if len(pixels.shape) == 4:
                        if pixels.shape[-1] == 4 or pixels.shape[1] == 4:
                            return True
            return False
        except Exception:
            return False

    def load(self, path: str, **kwargs) -> Dataset:
        keys_to_load = kwargs.get('keys_to_load', ['voxel', 'action'])
        transform = kwargs.get('transform', None)
        num_steps = kwargs.get('num_steps', 1)
        rgb_key = kwargs.get('rgb_key', self.rgb_key)
        depth_key = kwargs.get('depth_key', self.depth_key)
        voxel_key = kwargs.get('voxel_key', self.voxel_key)
        
        return HDF5RGBDVoxelDataset(
            path,
            keys_to_load,
            num_steps,
            transform,
            rgb_key,
            depth_key,
            voxel_key,
        )

    def get_input_channels(self) -> int:
        """体素是 3 通道：端点/自由空间/已观察"""
        return 3

    def get_transform(self, img_size):
        """返回体素转换 transform"""
        return VoxelizeTransform()

    def get_column_names(self) -> list:
        return ['voxel', 'pixels', 'depth', 'action', 'proprio', 'state']

    def supports_depth(self) -> bool:
        return True

    def get_depth_column(self) -> str:
        return self.depth_key

    def get_name(self) -> str:
        return "HDF5-RGBD-Voxel"