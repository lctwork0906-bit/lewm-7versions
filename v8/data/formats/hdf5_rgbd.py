"""
直接从HDF5读取预处理好的RGB-D数据
"""
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from ..base import DataFormatStrategy


class HDF5RGBDDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        with h5py.File(h5_path, 'r') as f:
            self.length = f['pixels'].shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            pixels = torch.from_numpy(f['pixels'][idx]).float()
            depth = torch.from_numpy(f['depth'][idx]).float()
            action = torch.tensor(f['action'][idx], dtype=torch.long)
            llm_tokens = torch.from_numpy(f['llm_tokens'][idx]).long()
        
        data = {
            'pixels': pixels,
            'depth': depth,
            'action': action,
            'llm_tokens': llm_tokens,
        }
        if self.transform:
            data = self.transform(data)
        return data


class HDF5RGBDStrategy(DataFormatStrategy):
    def detect(self, path):
        # 必须真的含有 RGB-D 像素/深度键，避免把"只有 voxel"的体素 HDF5 误判成 RGB-D
        # （否则 HDF5RGBDDataset 会因找不到 'pixels' 而 KeyError）。
        if not path.endswith('.h5'):
            return False
        try:
            with h5py.File(path, 'r') as f:
                return ('pixels' in f) and ('depth' in f or 'weight' in f)
        except Exception:
            return False
    
    def load(self, path, **kwargs):
        transform = kwargs.get('transform', None)
        return HDF5RGBDDataset(path, transform)
    
    def get_input_channels(self):
        return 4
    
    def get_transform(self, img_size):
        return None
    
    def get_column_names(self):
        return ['pixels', 'depth', 'action', 'llm_tokens']
    
    def get_name(self):
        return "HDF5-RGBD"
