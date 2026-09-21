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
        in_chans=4,
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
        pixels = batch['pixels']

        if pixels.dim() == 5:
            pixels = pixels.mean(dim=1)
        elif pixels.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected pixels shape: {pixels.shape}")

        emb = self.encoder(pixels)

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
