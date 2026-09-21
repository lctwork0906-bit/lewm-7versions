"""
JEPA 模型（支持 timm ViT 和多帧输入）
"""
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


class JEPA(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        pixels = info['pixels'].float()  # (B, T, C, H, W)
        B, T, C, H, W = pixels.shape
        
        # 展平时间和 batch: (B*T, C, H, W)
        pixels_flat = rearrange(pixels, "b t c h w -> (b t) c h w")
        
        # timm ViT 输出: (B*T, num_patches, D) 或 (B*T, D)
        output = self.encoder(pixels_flat)
        
        if isinstance(output, torch.Tensor):
            if output.dim() == 3:
                # (B*T, num_patches, D) -> 取 CLS token
                pixels_emb = output[:, 0]  # (B*T, D)
            else:
                pixels_emb = output
        else:
            pixels_emb = output.last_hidden_state[:, 0]
        
        # 恢复 batch 和时间维度: (B, T, D)
        emb = rearrange(pixels_emb, "(b t) d -> b t d", b=B, t=T)
        info["emb"] = emb
        
        if "action" in info:
            # action: (B, T, D) 已经是多帧
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
        return {"pred_loss": pred_loss, "emb": emb}
