"""
VLA-Simple: 3D-JEPA latent + 动作头（无LLM）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .action_head import ActionHead


class VLASimple(nn.Module):
    def __init__(
        self,
        jepa_encoder,
        jepa_predictor,
        action_encoder,
        projector,
        pred_proj,
        latent_dim=64,
        num_actions=8,
        dropout=0.3,
    ):
        super().__init__()
        self.jepa_encoder = jepa_encoder
        self.jepa_predictor = jepa_predictor
        self.action_encoder = action_encoder
        self.projector = projector
        self.pred_proj = pred_proj

        self.action_head = ActionHead(latent_dim, num_actions, dropout)

    def encode_3d(self, voxel):
        if voxel.dim() == 4:
            voxel = voxel.unsqueeze(1)
        elif voxel.dim() == 5:
            voxel = voxel.unsqueeze(1)
        else:
            while voxel.dim() < 6:
                voxel = voxel.unsqueeze(0)

        B, T, C, Z, Y, X = voxel.shape
        flat = rearrange(voxel, "b t c z y x -> (b t) c z y x")
        emb = self.jepa_encoder(flat)
        emb = self.projector(emb)
        emb = rearrange(emb, "(b t) d -> b t d", b=B, t=T)
        return emb

    def forward(self, batch):
        B = batch["voxel"].shape[0]
        device = batch["voxel"].device

        jepa_emb = self.encode_3d(batch["voxel"])
        T = jepa_emb.shape[1]

        # JEPA 预测损失
        if T > 1:
            ctx_emb = jepa_emb[:, :T-1]
            action_seq = batch.get("action_sequence", torch.zeros(B, T-1, 8, device=device))
            ctx_act = self.action_encoder(action_seq)
            pred_emb = self.jepa_predictor(ctx_emb, ctx_act)
            pred_emb = self.pred_proj(rearrange(pred_emb, "b t d -> (b t) d"))
            pred_emb = rearrange(pred_emb, "(b t) d -> b t d", b=B, t=T-1)
            pred_loss = F.mse_loss(pred_emb, jepa_emb[:, 1:, :])
        else:
            pred_loss = torch.tensor(0.0, device=device)

        # 动作预测
        action_pred = self.action_head(jepa_emb)

        # 动作损失
        action_loss = torch.tensor(0.0, device=device)
        if "action" in batch:
            action_target = batch["action"].squeeze(-1) if batch["action"].dim() > 1 else batch["action"]
            action_loss = F.cross_entropy(action_pred, action_target.long())

        total_loss = pred_loss + action_loss

        return {
            "losses": {
                "pred_loss": pred_loss,
                "action_loss": action_loss,
                "total_loss": total_loss,
            },
            "action_pred": action_pred,
            "jepa_emb": jepa_emb,
            "emb": jepa_emb,
        }
