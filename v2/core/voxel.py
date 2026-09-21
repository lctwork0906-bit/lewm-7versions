import math
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from typing import Dict, Tuple


class VoxelSpec:
    def __init__(self, z_cells=16, y_cells=48, x_cells=48, voxel_size=0.5, max_depth=20.0):
        self.z_cells = z_cells
        self.y_cells = y_cells
        self.x_cells = x_cells
        self.voxel_size = voxel_size
        self.max_depth = max_depth


class RGBDVoxelizer:
    def __init__(self, spec):
        self.spec = spec
        self.z_min = -4.0

    def _index(self, point):
        x, y, z = point
        ix = int(math.floor(x / self.spec.voxel_size + self.spec.x_cells / 2))
        iy = int(math.floor(y / self.spec.voxel_size + self.spec.y_cells / 2))
        iz = int(math.floor((z - self.z_min) / self.spec.voxel_size))
        if not (0 <= ix < self.spec.x_cells and 0 <= iy < self.spec.y_cells and 0 <= iz < self.spec.z_cells):
            return None
        return iz, iy, ix

    def build(self, rgb, depth):
        grid = np.zeros((3, self.spec.z_cells, self.spec.y_cells, self.spec.x_cells), dtype=np.float32)
        endpoints = []
        height, width = depth.shape
        samples = 12
        ys = np.linspace(0, height-1, samples).astype(int)
        xs = np.linspace(0, width-1, samples).astype(int)

        for iy in ys:
            v = (iy + 0.5 - height/2) / max(height/2, 1.0)
            for ix in xs:
                val = float(depth[iy, ix])
                if not math.isfinite(val) or val < 0.4:
                    continue
                u = (ix + 0.5 - width/2) / max(width/2, 1.0)
                ray = np.array([1.0, u, v], dtype=np.float32)
                ray = ray / max(np.linalg.norm(ray), 1e-6)
                visible = min(val, self.spec.max_depth)

                for frac in (0.25, 0.50, 0.75):
                    idx = self._index(ray * visible * frac)
                    if idx is not None:
                        grid[1, idx[0], idx[1], idx[2]] = 1.0
                        grid[2, idx[0], idx[1], idx[2]] = 1.0

                if val <= self.spec.max_depth:
                    endpoint = ray * val
                    idx = self._index(endpoint)
                    if idx is not None:
                        grid[0, idx[0], idx[1], idx[2]] = 1.0
                        grid[2, idx[0], idx[1], idx[2]] = 1.0
                        endpoints.append(endpoint)

        center = self._index((0.0, 0.0, 0.0))
        if center is not None:
            grid[1, center[0], center[1], center[2]] = 1.0
            grid[2, center[0], center[1], center[2]] = 1.0

        return torch.from_numpy(grid).float(), endpoints

    def collision_risk(self, occupied: dict, pose, radius=1.0):
        """
        碰撞检测：检查给定位置是否有碰撞风险
        occupied: dict {(x,y,z): count} 或 Counter
        pose: (x, y, z) 世界坐标
        radius: 检测半径 (体素格数)
        返回: 0.0 ~ 1.0 碰撞风险
        """
        center = self._index(pose[:3])
        if center is None:
            return 0.0
        hits = 0
        r = int(radius)
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                for dz in range(-r, r+1):
                    idx = (center[0]+dz, center[1]+dy, center[2]+dx)
                    if occupied.get(idx, 0) > 0:
                        hits += 1
        return min(1.0, hits / 4.0)

    def world_key(self, point, size=None):
        """将世界坐标转为体素索引键"""
        if size is None:
            size = self.spec.voxel_size
        return tuple(int(round(float(value) / size)) for value in point)


class VoxelJEPAEncoder(nn.Module):
    def __init__(self, spec, latent_dim=64):
        super().__init__()
        self.spec = spec
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 12, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((2, 3, 3)),
            nn.Flatten(),
            nn.Linear(24 * 2 * 3 * 3, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, voxel):
        if voxel.dim() == 4:
            voxel = voxel.unsqueeze(1)
            voxel = voxel.repeat(1, 3, 1, 1, 1)
        elif voxel.dim() == 5 and voxel.shape[1] != 3:
            if voxel.shape[1] == 1:
                voxel = voxel.repeat(1, 3, 1, 1, 1)
            else:
                voxel = voxel[:, :3, :, :, :]
        return self.encoder(voxel.float())


class CollisionAwarePlanner:
    """
    碰撞感知规划器
    在规划动作时考虑碰撞风险
    """
    def __init__(self, config):
        self.config = config
        self.occupied = Counter()  # 累积占用地图
        self.visited = Counter()   # 访问记录

    def update_occupancy(self, endpoints, position, yaw):
        """更新占用地图"""
        c, s = math.cos(yaw), math.sin(yaw)
        for endpoint in endpoints:
            world = position + np.asarray(
                [c * endpoint[0] - s * endpoint[1],
                 s * endpoint[0] + c * endpoint[1],
                 endpoint[2]]
            )
            key = tuple(int(round(v / self.config.voxel_size)) for v in world)
            self.occupied[key] += 1

    def collision_risk(self, pose, voxelizer, radius=1.0):
        """计算碰撞风险"""
        return voxelizer.collision_risk(self.occupied, pose, radius)

    def get_safe_actions(self, pose, voxelizer, actions, safety_threshold=0.3):
        """过滤出安全的动作"""
        safe = []
        for action in actions:
            next_pose = self._apply_action(pose, action)
            risk = self.collision_risk(next_pose, voxelizer)
            if risk < safety_threshold:
                safe.append((action, risk))
        return safe

    def _apply_action(self, pose, action):
        """应用动作，返回新位置"""
        x, y, z, yaw = pose
        step = self.config.get('step_size', 1.0)
        if action == 'forward':
            x += math.cos(yaw) * step
            y += math.sin(yaw) * step
        elif action == 'left':
            yaw -= math.radians(30)
        elif action == 'right':
            yaw += math.radians(30)
        elif action == 'back':
            x -= math.cos(yaw) * step
            y -= math.sin(yaw) * step
        return (x, y, z, yaw)

    def reset(self):
        """重置状态"""
        self.occupied.clear()
        self.visited.clear()
