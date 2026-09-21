import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from ..base import DataFormatStrategy
from ..transforms.voxelize import VoxelizeTransform

ACTION_MAP = {
    'forward': 0,
    'left': 1,
    'right': 2,
    'descend': 3,
    'ascend': 4,
    'rotl': 5,
    'rotr': 6,
    'stop': 7,
}


class PNGDepthDataset(Dataset):
    def __init__(self, root_dir, num_steps=1, transform=None, tokenizer=None, max_length=64):
        self.root_dir = root_dir
        self.num_steps = num_steps
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._load_metadata()

    def _load_metadata(self):
        self.tasks = []
        json_files = [f for f in os.listdir(self.root_dir) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files in {self.root_dir}")

        for json_path in json_files:
            json_path = os.path.join(self.root_dir, json_path)
            try:
                with open(json_path, 'r') as f:
                    content = f.read()
                    for line in content.splitlines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line.strip(), strict=False)
                            if 'steps' in data and data['steps']:
                                scene = data.get('map_name', 'unknown')
                                description = data.get('description', '')
                                for step_idx, step in enumerate(data['steps']):
                                    if 'rgb' in step and 'front' in step['rgb']:
                                        rgb_rel = step['rgb']['front']
                                        if rgb_rel.startswith('DATA/collected_vla/'):
                                            rgb_rel = rgb_rel[len('DATA/collected_vla/'):]
                                        rgb_path = os.path.join(self.root_dir, rgb_rel)
                                        if os.path.exists(rgb_path):
                                            self.tasks.append({
                                                'scene': scene,
                                                'episode_id': data.get('episode_id', 'unknown'),
                                                'step_idx': step_idx,
                                                'action': step.get('action', 'stop'),
                                                'description': description,
                                                'rgb': step['rgb'],
                                                'depth': step.get('depth', {}),
                                                'position': step.get('position', [0, 0, 0]),
                                                'quaternion': step.get('quaternion', [0, 0, 0, 0]),
                                                'frame': step.get('frame', step_idx),
                                            })
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Warning: Failed to parse {json_path}: {e}")

        print(f'Found {len(self.tasks)} valid RGB-D samples with actions')

    def __len__(self):
        return len(self.tasks)

    def _get_path(self, rel_path):
        if rel_path.startswith('DATA/collected_vla/'):
            rel_path = rel_path[len('DATA/collected_vla/'):]
        return os.path.join(self.root_dir, rel_path)

    def _compute_collision_risk(self, task):
        depth_path = None
        if 'depth' in task and 'front' in task['depth']:
            depth_rel = task['depth']['front']
            depth_path = self._get_path(depth_rel)
        if depth_path and os.path.exists(depth_path):
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is not None:
                mean_depth = np.mean(depth) / 1000.0
                risk = 1.0 - min(mean_depth / 5.0, 1.0)
                return torch.tensor(risk).float()
        return torch.tensor(0.0).float()

    def __getitem__(self, idx):
        task = self.tasks[idx]

        rgb_path = self._get_path(task['rgb']['front'])
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            for angle in ['front', 'left', 'right', 'down']:
                if angle in task['rgb']:
                    rgb_path = self._get_path(task['rgb'][angle])
                    rgb = cv2.imread(rgb_path)
                    if rgb is not None:
                        break

        if rgb is None:
            return self.__getitem__((idx + 1) % len(self.tasks))

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = None
        if 'depth' in task and 'front' in task['depth']:
            depth_path = self._get_path(task['depth']['front'])
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if depth is None:
            depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        else:
            depth = depth.astype(np.float32)

        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        depth = torch.from_numpy(depth).float()

        action_str = task.get('action', 'stop')
        action_idx = ACTION_MAP.get(action_str, ACTION_MAP['stop'])
        action = torch.tensor([action_idx], dtype=torch.long)

        collision_risk = self._compute_collision_risk(task)

        # 语言指令：总是返回 tensor，不返回 None
        description = task.get('description', '')
        if self.tokenizer is not None and description:
            tokens = self.tokenizer(
                description,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            llm_tokens = tokens['input_ids'].squeeze(0)
        else:
            # 没有 description 或 tokenizer 时，用零填充
            llm_tokens = torch.zeros(self.max_length, dtype=torch.long)

        data = {
            'pixels': rgb,
            'depth': depth,
            'action': action,
            'collision_risk': collision_risk,
            'description': description,
            'llm_tokens': llm_tokens,
        }

        if self.transform:
            data = self.transform(data)

        return data


class PNGDepthStrategy(DataFormatStrategy):
    def detect(self, path: str) -> bool:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith('.json'):
                    return True
        return False

    def load(self, path: str, **kwargs) -> Dataset:
        transform = kwargs.get('transform', None)
        tokenizer = kwargs.get('tokenizer', None)
        max_length = kwargs.get('max_length', 64)
        return PNGDepthDataset(path, transform=transform, tokenizer=tokenizer, max_length=max_length)

    def get_input_channels(self) -> int:
        return 3

    def get_transform(self, img_size):
        return VoxelizeTransform()

    def get_column_names(self) -> list:
        return ['pixels', 'depth', 'action', 'collision_risk', 'description', 'llm_tokens']

    def get_name(self) -> str:
        return "PNG-RGBD"
