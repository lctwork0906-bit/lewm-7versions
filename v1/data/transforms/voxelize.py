import torch
from core.voxel import RGBDVoxelizer, VoxelSpec

class VoxelizeTransform:
    def __init__(self, spec=None):
        self.spec = spec or VoxelSpec()
        self.voxelizer = RGBDVoxelizer(self.spec)
    
    def __call__(self, data):
        if 'rgb' in data and 'depth' in data:
            rgb = data['rgb']
            depth = data['depth']
        elif 'pixels' in data and 'depth' in data:
            rgb = data['pixels']
            depth = data['depth']
        else:
            raise KeyError(f"Need 'rgb' and 'depth' or 'pixels' and 'depth', got {list(data.keys())}")
        
        # 如果 depth 是 4 视角堆叠 (4, H, W)
        if depth.dim() == 3 and depth.shape[0] == 4:
            voxels = []
            for i in range(4):
                v, _ = self.voxelizer.build(
                    rgb[i].permute(1, 2, 0).cpu().numpy().astype('uint8'),
                    depth[i].cpu().numpy()
                )
                voxels.append(v)
            voxel = torch.stack(voxels, dim=0).sum(dim=0)
            voxel = torch.clamp(voxel, 0, 1)
        else:
            voxel, _ = self.voxelizer.build(
                rgb.permute(1, 2, 0).cpu().numpy().astype('uint8'),
                depth.cpu().numpy()
            )
        
        # 确保体素是 3 通道 (端点/自由/观察)
        # voxel 是 (3, Z, Y, X)，已经是 3 通道
        # 但可能因为 sum 操作变成了 (Z, Y, X)
        if voxel.dim() == 3:
            # (Z, Y, X) -> (1, Z, Y, X)，需要把通道数扩展到 3
            # 但更好的方式是用 3 通道
            voxel = voxel.unsqueeze(0)
            # 如果只有一个通道，复制到 3 个通道
            voxel = voxel.repeat(3, 1, 1, 1)
        
        data['voxel'] = voxel
        return data
