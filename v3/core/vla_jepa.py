"""
VLA-JEPA: 3D-JEPA + LLM + Cross-Attention 联合模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel

from .cross_attention import CrossAttentionModule, SpatialTokenEncoder


class VLAJEPA(nn.Module):
    def __init__(
        self,
        jepa_encoder,
        jepa_predictor,
        action_encoder,
        projector,
        pred_proj,
        llm_model_name: str = "gpt2",
        llm_dim: int = 128,
        jepa_dim: int = 64,
        num_heads: int = 8,
        use_adaln: bool = True,
        freeze_llm: bool = True,
    ):
        super().__init__()
        
        self.jepa_encoder = jepa_encoder
        self.jepa_predictor = jepa_predictor
        self.action_encoder = action_encoder
        self.projector = projector
        self.pred_proj = pred_proj
        
        self.llm = AutoModel.from_pretrained(llm_model_name)
        self.llm_dim = llm_dim
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
        
        self.spatial_tokenizer = SpatialTokenEncoder(jepa_dim, llm_dim)
        self.cross_attn = CrossAttentionModule(jepa_dim, llm_dim, num_heads, use_adaln=use_adaln)
        self.action_head = nn.Sequential(
            nn.Linear(llm_dim, 256),
            nn.GELU(),
            nn.Linear(256, 8),
        )
        self.loss_weights = {'pred': 1.0, 'action': 1.0}
    
    def encode_3d(self, voxel):
        if voxel.dim() == 4:
            voxel = voxel.unsqueeze(1)
        elif voxel.dim() == 5:
            voxel = voxel.unsqueeze(1)
            pass
        elif voxel.dim() == 3:
            voxel = voxel.unsqueeze(0).unsqueeze(0)
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
        B = batch['voxel'].shape[0]
        device = batch['voxel'].device
        
        jepa_emb = self.encode_3d(batch['voxel'])
        T = jepa_emb.shape[1]
        
        if T > 1:
            ctx_emb = jepa_emb[:, :T-1]
            action_seq = batch.get('action_sequence', torch.zeros(B, T-1, 8, device=device))
            ctx_act = self.action_encoder(action_seq)
            pred_emb = self.jepa_predictor(ctx_emb, ctx_act)
            pred_emb = self.pred_proj(rearrange(pred_emb, "b t d -> (b t) d"))
            pred_emb = rearrange(pred_emb, "(b t) d -> b t d", b=B, t=T-1)
            pred_loss = F.mse_loss(pred_emb, jepa_emb[:, 1:, :])
        else:
            pred_loss = torch.tensor(0.0, device=device)
        
        spatial_tokens = self.spatial_tokenizer(jepa_emb)
        llm_tokens = torch.zeros(B, 1, self.llm_dim, device=device)
        fused_tokens, attn_weights = self.cross_attn(llm_tokens, spatial_tokens)
        
        action_pred = self.action_head(fused_tokens.mean(dim=1))
        
        action_loss = torch.tensor(0.0, device=device)
        if 'action' in batch:
            action_target = batch['action'].squeeze(-1) if batch['action'].dim() > 1 else batch['action']
            action_loss = F.cross_entropy(action_pred, action_target.long())
        
        total_loss = self.loss_weights['pred'] * pred_loss + self.loss_weights['action'] * action_loss
        
        return {
            'losses': {'pred_loss': pred_loss, 'action_loss': action_loss, 'total_loss': total_loss},
            'action_pred': action_pred,
            'jepa_emb': jepa_emb,
            'emb': jepa_emb,
        }
