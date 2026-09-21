"""
3D 体素 JEPA 模型
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
        
        # 确保体素是 3 通道 (B, C, Z, Y, X) 或 (B, T, C, Z, Y, X)
        if voxel.dim() == 4:
            voxel = voxel.unsqueeze(1)  # (B, 1, Z, Y, X)
            voxel = voxel.repeat(1, 3, 1, 1, 1)  # (B, 3, Z, Y, X)
        elif voxel.dim() == 5:
            if voxel.shape[1] != 3:
                if voxel.shape[1] == 1:
                    voxel = voxel.repeat(1, 3, 1, 1, 1)
                else:
                    voxel = voxel[:, :3, :, :, :]
        elif voxel.dim() == 6:
            # (B, T, C, Z, Y, X) -> 展平时间和batch
            B, T, C, Z, Y, X = voxel.shape
            voxel = rearrange(voxel, "b t c z y x -> (b t) c z y x")
        else:
            raise ValueError(f"Unexpected voxel shape: {voxel.shape}")
        
        # 编码
        emb = self.encoder(voxel)
        emb = self.projector(emb)
        
        # 恢复时间维度（如果有）
        if voxel.dim() == 6:
            emb = rearrange(emb, "(b t) d -> b t d", b=B, t=T)
            info["emb"] = emb
        else:
            info["emb"] = emb.unsqueeze(1)  # (B, 1, D)
        
        if "action" in info:
            action = info["action"]
            # 确保 action 是 3 维 (B, T, D)
            if action.dim() == 2:
                action = action.unsqueeze(1)  # (B, 1, D)
            info["act_emb"] = self.action_encoder(action)
        
        return info

    def predict(self, emb, act_emb):
        return self.predictor(emb, act_emb)

    def forward(self, batch):
        output = self.encode(batch)
        emb = output["emb"]
        act_emb = output["act_emb"]
        
        T = emb.shape[1]
        if T > 1:
            ctx_emb = emb[:, :T-1]
            ctx_act = act_emb[:, :T-1]
            tgt_emb = emb[:, 1:]
            pred_emb = self.predict(ctx_emb, ctx_act)
            pred_loss = F.mse_loss(pred_emb, tgt_emb)
        else:
            # 单帧时用自回归预测
            pred_emb = self.predict(emb, act_emb)
            pred_loss = F.mse_loss(pred_emb, emb.detach())
        
        return {"pred_loss": pred_loss, "emb": emb}
