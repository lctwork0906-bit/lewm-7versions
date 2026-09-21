"""
RGBD-JEPA: RGB-D 图像预测下一帧 RGB-D
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


class RGBDJEPA(nn.Module):
    def __init__(
        self,
        img_size=512,
        in_chans=4,
        embed_dim=192,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=0,
            in_chans=in_chans,
            img_size=img_size,
        )
        self.embed_dim = embed_dim

        # 预测头：从 latent 预测下一帧 latent
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, batch):
        pixels = batch['pixels']

        if pixels.dim() == 5:
            pixels = pixels.mean(dim=1)
        elif pixels.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected pixels shape: {pixels.shape}")

        # 编码当前帧
        emb = self.encoder(pixels)  # (B, embed_dim)

        # 预测下一帧 latent（自回归）
        pred_emb = self.predictor(emb)

        # 损失：MSE 预测损失
        pred_loss = F.mse_loss(pred_emb, emb.detach())

        return {
            "losses": {
                "pred_loss": pred_loss,
                "total_loss": pred_loss,
            },
            "pred_emb": pred_emb,
            "emb": emb,
        }
