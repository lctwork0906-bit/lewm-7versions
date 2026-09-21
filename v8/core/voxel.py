import math
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from typing import Dict, Tuple


class VoxelSpec:
    """
    体素网格规格。

    - 分辨率 (z_cells, y_cells, x_cells) 与体素边长 voxel_size(米) 决定覆盖空间：
        每轴半幅 = cells * voxel_size / 2。
        默认 16×48×48 @ 0.5m -> X/Y 覆盖 ±12m，Z 覆盖 [z_min, z_min+8m]。
    - use_color=True 时，体素从 3 通道(几何) 升级为 6 通道：
        [占据(端点), 自由空间, 归一化深度, R, G, B]
        这样 RGB-D 的"颜色/语义"信息被保留进体素（修复此前 build 丢弃 rgb 的问题）。
    - samples 控制投射线密度（原 12 -> 12×12=144 条，太稀疏）。
    """
    def __init__(self, z_cells=16, y_cells=48, x_cells=48, voxel_size=0.5,
                 max_depth=20.0, z_min=-4.0, use_color=False, samples=12):
        self.z_cells = z_cells
        self.y_cells = y_cells
        self.x_cells = x_cells
        self.voxel_size = voxel_size
        self.max_depth = max_depth
        self.z_min = z_min
        self.use_color = use_color
        self.samples = samples

    @property
    def in_channels(self):
        """通道数：开颜色=6，否则=3（几何）。"""
        return 6 if self.use_color else 3


class RGBDVoxelizer:
    def __init__(self, spec):
        self.spec = spec
        self.z_min = getattr(spec, 'z_min', -4.0)

    def _index(self, point):
        x, y, z = point
        ix = int(math.floor(x / self.spec.voxel_size + self.spec.x_cells / 2))
        iy = int(math.floor(y / self.spec.voxel_size + self.spec.y_cells / 2))
        iz = int(math.floor((z - self.z_min) / self.spec.voxel_size))
        if not (0 <= ix < self.spec.x_cells and 0 <= iy < self.spec.y_cells and 0 <= iz < self.spec.z_cells):
            return None
        return iz, iy, ix

    def build(self, rgb, depth):
        """
        将一帧 RGB-D 反投影为 3D 占据栅格。

        rgb : (H, W, 3) uint8/float，或 None（不填颜色）
        depth: (H, W) 米

        返回 (C, Z, Y, X) 体素张量，C=3(几何) 或 6(几何+深度+RGB)。

        通道布局：
          use_color=False (3ch, 向后兼容):
            C0 = 端点/表面占据, C1 = 自由空间(射线经过), C2 = 两者并集
          use_color=True  (6ch):
            C0 = 端点/表面占据, C1 = 自由空间, C2 = 归一化深度(val/max_depth),
            C3/C4/C5 = 表面点的 R/G/B (0~1)
        """
        use_color = getattr(self.spec, 'use_color', False)
        n_samples = int(getattr(self.spec, 'samples', 12))
        C = 6 if use_color else 3
        grid = np.zeros((C, self.spec.z_cells, self.spec.y_cells, self.spec.x_cells), dtype=np.float32)
        endpoints = []
        height, width = depth.shape

        # 预备 RGB（若启用颜色且 rgb 可用）
        rgb_arr = None
        if use_color and rgb is not None:
            rgb_arr = np.asarray(rgb, dtype=np.float32)
            if rgb_arr.ndim == 3 and rgb_arr.shape[0] == 3:   # (3,H,W) -> (H,W,3)
                rgb_arr = rgb_arr.transpose(1, 2, 0)
            if rgb_arr.ndim == 2:                              # 灰度，无法取色
                rgb_arr = None
            elif rgb_arr.shape[-1] > 3:                       # RGBA 等，仅取前 3 通道
                rgb_arr = rgb_arr[..., :3]
        if rgb_arr is not None and rgb_arr.dtype != np.float32:
            rgb_arr = rgb_arr.astype(np.float32)

        ys = np.linspace(0, height - 1, n_samples).astype(int)
        xs = np.linspace(0, width - 1, n_samples).astype(int)

        for iy in ys:
            v = (iy + 0.5 - height / 2) / max(height / 2, 1.0)
            for ix in xs:
                val = float(depth[iy, ix])
                if not math.isfinite(val) or val < 0.4:
                    continue
                u = (ix + 0.5 - width / 2) / max(width / 2, 1.0)
                ray = np.array([1.0, u, v], dtype=np.float32)
                ray = ray / max(np.linalg.norm(ray), 1e-6)
                visible = min(val, self.spec.max_depth)

                # 自由空间：射线经过的体素
                for frac in (0.25, 0.50, 0.75):
                    idx = self._index(ray * visible * frac)
                    if idx is not None:
                        grid[1, idx[0], idx[1], idx[2]] = 1.0
                        if not use_color:
                            grid[2, idx[0], idx[1], idx[2]] = 1.0

                # 表面端点
                if val <= self.spec.max_depth:
                    endpoint = ray * val
                    idx = self._index(endpoint)
                    if idx is not None:
                        grid[0, idx[0], idx[1], idx[2]] = 1.0
                        if use_color:
                            grid[2, idx[0], idx[1], idx[2]] = min(val / self.spec.max_depth, 1.0)
                            if rgb_arr is not None:
                                r, g, b = rgb_arr[iy, ix][:3]
                                grid[3, idx[0], idx[1], idx[2]] = r / 255.0
                                grid[4, idx[0], idx[1], idx[2]] = g / 255.0
                                grid[5, idx[0], idx[1], idx[2]] = b / 255.0
                        else:
                            grid[2, idx[0], idx[1], idx[2]] = 1.0
                        endpoints.append(endpoint)

        center = self._index((0.0, 0.0, 0.0))
        if center is not None:
            grid[1, center[0], center[1], center[2]] = 1.0
            if not use_color:
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
    """
    真正的 3D 卷积骨干编码器。

    关键改变（修复"压成 18 个位置"的智障操作）：
      - 不再用 AdaptiveAvgPool3d((2,3,3)) 把整张体素碾成固定 2×3×3=18 个位置，
        而是用逐 stage stride-2 的 3D 卷积下采样，保留与输入分辨率成正比的
        空间特征图 -> 输出 N = zf·yf·xf 个空间 token。
      - 分辨率越高，token 越多、越细（16×48×48 -> N≈72；32×96×96 -> N≈576），
        "分辨率"从此真正有意义。
      - forward(voxel) 仍返回 (B, latent_dim) 单向量（给 voxel / vla_simple 的
        世界模型分支 / 动作头用），保持下游维度不变；
      - encode(voxel) / forward_tokens(voxel) 返回 (B, N, token_dim) 空间 token，
        供 vla_jepa 的 cross-attention 真正做"语言查询 3D 视觉"。
    """

    def __init__(self, spec, latent_dim=64, token_dim=128, num_stages=3, base_channels=32):
        super().__init__()
        self.spec = spec
        self.latent_dim = latent_dim
        self.token_dim = token_dim
        self.num_stages = num_stages
        self.in_channels = getattr(spec, 'in_channels', 3)

        # ---- 3D 卷积骨干：逐级下采样，保持空间结构 ----
        stages = []
        c_in = self.in_channels
        c = base_channels
        for _ in range(num_stages):
            stages.append(nn.Sequential(
                nn.Conv3d(c_in, c, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(c),
                nn.GELU(),
            ))
            c_in = c
            c *= 2
        self.backbone = nn.ModuleList(stages)
        self.backbone_out_channels = c_in

        # 1×1 卷积把骨干输出通道对齐到 token_dim
        self.token_proj = nn.Conv3d(self.backbone_out_channels, token_dim, kernel_size=1, bias=False)

        # token -> latent（给需要"单向量"的下游分支）
        self.pool_to_latent = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def _normalize_input(self, voxel):
        x = voxel.float()
        if x.dim() == 4:                       # (C,Z,Y,X) 单样本
            x = x.unsqueeze(0)
        elif x.dim() == 3:                      # (Z,Y,X) 单通道
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 6:                      # (B,T,C,Z,Y,X) 防御
            B, T, C, Z, Y, X = x.shape
            x = x.reshape(B * T, C, Z, Y, X)
        if x.shape[1] != self.in_channels:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_channels, 1, 1, 1)
            else:
                x = x[:, :self.in_channels, :, :, :]
        return x

    def _backbone_features(self, voxel):
        x = self._normalize_input(voxel)
        for stage in self.backbone:
            x = stage(x)
        x = self.token_proj(x)                  # (B, token_dim, zf, yf, xf)
        return x

    def encode(self, voxel):
        x = self._backbone_features(voxel)
        B, D, zf, yf, xf = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).reshape(B, zf * yf * xf, D)  # (B, N, token_dim)
        pooled = self.pool_to_latent(tokens.mean(dim=1))              # (B, latent_dim)
        return pooled, tokens

    def forward(self, voxel):
        """向后兼容：返回 (B, latent_dim) 单向量。"""
        pooled, _ = self.encode(voxel)
        return pooled

    def forward_tokens(self, voxel):
        """返回空间 token (B, N, token_dim)，N 随分辨率增长。"""
        _, tokens = self.encode(voxel)
        return tokens


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
