"""
3D 体素 JEPA 模型 + 碰撞感知
"""
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


class JEPA3D(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        voxel = info['voxel'].float()
        if len(voxel.shape) == 5:
            voxel = voxel.unsqueeze(2)
        B, T, C, D, H, W = voxel.shape

        voxel_flat = rearrange(voxel, "b t c d h w -> (b t) c d h w")
        emb = self.encoder(voxel_flat)
        emb = self.projector(emb)
        emb = rearrange(emb, "(b t) d -> b t d", b=B, t=T)
        info["emb"] = emb

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb, act_emb):
        return self.predictor(emb, act_emb)

    def forward(self, batch):
        output = self.encode(batch)
        emb = output["emb"]
        act_emb = output["act_emb"]
        T = emb.shape[1]
        ctx_emb = emb[:, :T-1]
        ctx_act = act_emb[:, :T-1]
        tgt_emb = emb[:, 1:]
        pred_emb = self.predict(ctx_emb, ctx_act)
        pred_loss = F.mse_loss(pred_emb, tgt_emb)

        # ========== 碰撞感知损失 ==========
        collision_loss = torch.tensor(0.0, device=emb.device)
        if "collision_risk" in batch:
            risk = batch["collision_risk"].float()
            # 鼓励预测的 embedding 的最后一个时间步对应低碰撞风险
            # 用 pred_emb 的最后一个时间步来预测碰撞风险
            pred_risk = torch.sigmoid(pred_emb[:, -1, 0])  # 简化：用第一个维度
            collision_loss = F.mse_loss(pred_risk, risk)
            # 也可以加一个正则项：直接约束 emb
            # collision_loss = F.mse_loss(pred_emb[:, -1], risk.unsqueeze(-1).expand(-1, pred_emb.shape[-1]))

        total_loss = pred_loss + 0.1 * collision_loss  # 碰撞损失权重 0.1

        return {"pred_loss": pred_loss, "collision_loss": collision_loss, "loss": total_loss, "emb": emb}
