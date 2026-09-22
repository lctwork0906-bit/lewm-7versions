import torch
from core.voxel import RGBDVoxelizer, VoxelSpec


class VoxelizeTransform:
    def __init__(self, spec=None):
        self.spec = spec or VoxelSpec()
        self.voxelizer = RGBDVoxelizer(self.spec)
        self.in_channels = self.spec.in_channels

    def __call__(self, data):
        # 数据已自带预计算体素（'voxel'），无需再 voxelize，直接透传。
        # 否则（仅带 raw pixels/depth 的文件）才执行 voxelization。
        if 'voxel' in data:
            return data
        if 'rgb' in data and 'depth' in data:
            rgb = data['rgb']
            depth = data['depth']
        elif 'pixels' in data and 'depth' in data:
            rgb = data['pixels']
            depth = data['depth']
        else:
            raise KeyError(f"Need 'rgb' and 'depth' or 'pixels' and 'depth', got {list(data.keys())}")

        # 若 depth 是 4 视角堆叠 (4, H, W)
        if depth.dim() == 3 and depth.shape[0] == 4:
            voxels = []
            for i in range(4):
                v, _ = self.voxelizer.build(
                    rgb[i].permute(1, 2, 0).cpu().numpy().astype('uint8'),
                    depth[i].cpu().numpy()
                )
                voxels.append(v)
            voxel = self._combine_views(voxels)
        else:
            voxel, _ = self.voxelizer.build(
                rgb.permute(1, 2, 0).cpu().numpy().astype('uint8'),
                depth.cpu().numpy()
            )

        data['voxel'] = voxel
        return data

    def _combine_views(self, voxels):
        """
        多视角体素合并（修复原 sum+clamp 对颜色通道的错误累加）。

        - 几何通道(0,1,2)：各视角取最大（存在即 1；深度取最远）。
        - 颜色通道(3,4,5)：仅对有颜色的视角取平均，避免 0 视图稀释。
        """
        stacked = torch.stack(voxels, dim=0)              # (V, C, Z, Y, X)
        C = stacked.shape[1]
        geo = stacked[:, :3].clamp(0, 1).max(dim=0).values
        if C > 3:
            color = stacked[:, 3:]                        # (V, 3, Z, Y, X)
            present = (color.sum(dim=1, keepdim=True) > 0).float()
            color_sum = color.sum(dim=0)
            present_sum = present.sum(dim=0).clamp(min=1.0)
            color_avg = color_sum / present_sum
            return torch.cat([geo, color_avg], dim=0)
        return geo
