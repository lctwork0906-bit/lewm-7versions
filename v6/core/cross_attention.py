"""
Cross-Attention模块：将3D-JEPA embedding与LLM token融合
"""
import torch
import torch.nn as nn


class CrossAttentionModule(nn.Module):
    def __init__(
        self,
        jepa_dim: int = 64,
        llm_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_adaln: bool = False,
    ):
        super().__init__()
        self.llm_dim = llm_dim

        self.query_proj = nn.Linear(llm_dim, llm_dim)
        self.key_proj = nn.Linear(llm_dim, llm_dim)
        self.value_proj = nn.Linear(llm_dim, llm_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, llm_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, llm_tokens: torch.Tensor, spatial_tokens: torch.Tensor):
        query = self.query_proj(llm_tokens)
        key = self.key_proj(spatial_tokens)
        value = self.value_proj(spatial_tokens)

        attn_output, attn_weights = self.cross_attn(
            query=query,
            key=key,
            value=value,
            need_weights=True,
        )

        fused = self.norm1(llm_tokens + self.dropout(attn_output))
        fused = self.norm2(fused + self.dropout(self.ffn(fused)))

        return fused, attn_weights


class SpatialTokenEncoder(nn.Module):
    def __init__(self, jepa_dim: int = 64, token_dim: int = 256):
        super().__init__()
        self.token_dim = token_dim
        self.projector = nn.Sequential(
            nn.Linear(jepa_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 8, token_dim) * 0.02)

    def forward(self, jepa_emb: torch.Tensor):
        if jepa_emb.dim() == 2:
            jepa_emb = jepa_emb.unsqueeze(1)
        B, T, D = jepa_emb.shape
        tokens = self.projector(jepa_emb)
        if T < self.pos_embed.shape[1]:
            tokens = tokens + self.pos_embed[:, :T, :]
        else:
            tokens = tokens + self.pos_embed
        return tokens
