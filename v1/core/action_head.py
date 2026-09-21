"""
动作头：直接用 3D-JEPA latent 预测动作
"""
import torch
import torch.nn as nn


class ActionHead(nn.Module):
    def __init__(self, latent_dim=64, num_actions=8, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_actions),
        )

    def forward(self, latent):
        if latent.dim() == 3:
            latent = latent.mean(dim=1)
        return self.head(latent)
