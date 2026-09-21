"""
RGB-D + LLM 动作分类（LLM 可训练）
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
        llm_model_name="distilgpt2",
        llm_dim=768,
        num_heads=4,
        freeze_llm=False,
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

        self.action_head = nn.Sequential(
            nn.Linear(llm_dim, 128),
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
        spatial_tokens = self.spatial_tokenizer(emb.unsqueeze(1))

        if 'llm_tokens' in batch and batch['llm_tokens'] is not None:
            llm_input_ids = batch['llm_tokens'].long()
            llm_tokens = self.llm.get_input_embeddings()(llm_input_ids)
        else:
            llm_tokens = torch.zeros(pixels.size(0), 1, self.llm_dim, device=pixels.device)

        fused_tokens, _ = self.cross_attn(llm_tokens, spatial_tokens)
        action_pred = self.action_head(fused_tokens.mean(dim=1))

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
