"""
RGB-D + LLM 动作分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from transformers import AutoModel
from .cross_attention import CrossAttentionModule, SpatialTokenEncoder


class RGBVLA(nn.Module):
    def __init__(
        self,
        img_size=512,
        in_chans=4,
        num_actions=8,
        dropout=0.3,
        llm_model_name="google/bert_uncased_L-2_H-128_A-2",
        llm_dim=128,
        num_heads=4,
        freeze_llm=True,
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

        self.llm = AutoModel.from_pretrained(llm_model_name)
        self.llm_dim = llm_dim
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        self.spatial_tokenizer = SpatialTokenEncoder(self.embed_dim, llm_dim)
        self.cross_attn = CrossAttentionModule(self.embed_dim, llm_dim, num_heads, use_adaln=False)
        
        # 投影层：128 → 192，让 fused_tokens 匹配 action_head 的输入
        self.fuse_proj = nn.Linear(llm_dim, self.embed_dim)
        
        # action_head: 和检查点一致 192 → 128 → 8
        self.action_head = nn.Sequential(
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_actions),
        )

    def forward(self, batch):
        pixels = batch['pixels']
        depth = batch['depth']

        B, n_views, C, H, W = pixels.shape

        pixels_flat = pixels.view(B * n_views, C, H, W)
        depth_flat = depth.view(B * n_views, 1, H, W)

        rgbd = torch.cat([pixels_flat, depth_flat], dim=1)

        emb = self.encoder(rgbd)
        emb = emb.view(B, n_views, self.embed_dim).mean(dim=1)

        spatial_tokens = self.spatial_tokenizer(emb.unsqueeze(1))

        if 'llm_tokens' in batch:
            llm_tokens = batch['llm_tokens']
            llm_tokens = self.llm.get_input_embeddings()(llm_tokens)
        else:
            llm_tokens = torch.zeros(pixels.size(0), 1, self.llm_dim, device=pixels.device)

        fused_tokens, _ = self.cross_attn(llm_tokens, spatial_tokens)

        # 投影到192，匹配 action_head 的输入
        fused = self.fuse_proj(fused_tokens.mean(dim=1))

        action_pred = self.action_head(fused)

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
