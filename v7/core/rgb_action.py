"""
RGB-D 直接动作分类（无体素，无LLM）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


class RGBDActionModel(nn.Module):
    def __init__(
        self,
        img_size=512,
        in_chans=4,  # RGB + Depth
        num_actions=8,
        dropout=0.3,
    ):
        super().__init__()
        self.encoder = create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=0,
            in_chans=in_chans,
            img_size=img_size,
        )
        self.embed_dim = 192

        self.action_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_actions),
        )

    def forward(self, batch):
        pixels = batch['pixels']  # (B, 4, 3, H, W) 4视角
        depth = batch['depth']    # (B, 4, H, W) 4视角

        # 合并 RGB 和 Depth 为 4 通道
        # 取 4 个视角的均值，然后合并
        # 或者直接拼接：每个视角 (3+H, W) -> 但 ViT 需要固定通道数
        # 方案：对 4 个视角分别处理，然后平均
        B, n_views, C, H, W = pixels.shape
        
        # 展平视角到 batch 维度
        pixels_flat = pixels.view(B * n_views, C, H, W)  # (B*4, 3, H, W)
        depth_flat = depth.view(B * n_views, 1, H, W)    # (B*4, 1, H, W)
        
        # 合并为 4 通道 (RGB + Depth)
        rgbd = torch.cat([pixels_flat, depth_flat], dim=1)  # (B*4, 4, H, W)
        
        # ViT 编码
        emb = self.encoder(rgbd)  # (B*4, 192)
        
        # 恢复视角维度，取平均
        emb = emb.view(B, n_views, self.embed_dim).mean(dim=1)  # (B, 192)
        
        # 动作预测
        action_pred = self.action_head(emb)

        action_loss = torch.tensor(0.0, device=pixels.device)
        if "action" in batch:
            action_target = batch["action"].squeeze(-1) if batch["action"].dim() > 1 else batch["action"]
            action_loss = F.cross_entropy(action_pred, action_target.long())

        return {
            "losses": {
                "action_loss": action_loss,
                "total_loss": action_loss,
            },
            "action_pred": action_pred,
            "emb": emb,
        }
