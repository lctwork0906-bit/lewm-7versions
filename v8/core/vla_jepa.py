"""
VLA-JEPA: 3D-JEPA + LLM + Cross-Attention 联合模型
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel

from .cross_attention import CrossAttentionModule


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
        jepa_token_dim: int = 128,
        num_heads: int = 8,
        use_adaln: bool = True,
        freeze_llm: bool = True,
        num_actions: int = 8,
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

        # 把 3D-JEPA 的空间 token (token_dim) 投影到 LLM 空间 (llm_dim) 供 cross-attn
        self.token_adapter = nn.Linear(jepa_token_dim, llm_dim)
        # 可学习的 query token：替代原来的全零 query（全零会让注意力塌成均匀池化）
        self.query_token = nn.Parameter(torch.zeros(1, 1, llm_dim))
        self.cross_attn = CrossAttentionModule(jepa_dim, llm_dim, num_heads, use_adaln=use_adaln)
        self.num_actions = num_actions
        self.action_head = nn.Sequential(
            nn.Linear(llm_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_actions),
        )
        # 语言投影：把冻结 LLM 的 hidden 维映射到 llm_dim，作为 cross-attn 的 query。
        # 若 LLM 没有 config（某些 stub / 特例），退化为 llm_dim -> llm_dim。
        llm_hidden = getattr(getattr(self.llm, "config", None), "hidden_size", llm_dim)
        self.lang_proj = nn.Linear(llm_hidden, llm_dim)
        self.loss_weights = {'pred': 1.0, 'action': 1.0}

    @staticmethod
    def _sinusoid_pe(n: int, dim: int, device):
        """标准 Transformer 正弦位置编码（按 token 序号，分辨率自适应）。"""
        pos = torch.arange(n, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(n, dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def encode_3d(self, voxel):
        if voxel.dim() == 4:
            voxel = voxel.unsqueeze(1)
        elif voxel.dim() == 5:
            voxel = voxel.unsqueeze(1)
        elif voxel.dim() == 3:
            voxel = voxel.unsqueeze(0).unsqueeze(0)
        else:
            while voxel.dim() < 6:
                voxel = voxel.unsqueeze(0)

        B, T, C, Z, Y, X = voxel.shape
        flat = rearrange(voxel, "b t c z y x -> (b t) c z y x")
        # 同时拿到"单向量"(给 JEPA 预测) 与"空间 token"(给 cross-attn)
        jepa_emb, tokens = self.jepa_encoder.encode(flat)
        jepa_emb = rearrange(jepa_emb, "(b t) d -> b t d", b=B, t=T)
        return jepa_emb, tokens, B, T

    def forward(self, batch):
        B = batch['voxel'].shape[0]
        device = batch['voxel'].device

        jepa_emb, tokens, B, T = self.encode_3d(batch['voxel'])

        # ---- JEPA 预测损失（世界模型分支，不变）----
        if T > 1:
            ctx_emb = jepa_emb[:, :T - 1]
            action_seq = batch.get('action_sequence', torch.zeros(B, T - 1, 8, device=device))
            ctx_act = self.action_encoder(action_seq)
            pred_emb = self.jepa_predictor(ctx_emb, ctx_act)
            pred_emb = self.pred_proj(rearrange(pred_emb, "b t d -> (b t) d"))
            pred_emb = rearrange(pred_emb, "(b t) d -> b t d", b=B, t=T - 1)
            pred_loss = F.mse_loss(pred_emb, jepa_emb[:, 1:, :])
        else:
            pred_loss = torch.tensor(0.0, device=device)

        # ---- 空间 token -> cross-attention（分辨率真正生效）----
        spatial = self.token_adapter(tokens)                 # (B*T, N, llm_dim)
        N = spatial.shape[1]
        spatial = spatial + self._sinusoid_pe(N, self.llm_dim, device)

        # query：优先用语言（语义目标，如 "fly to object_5029"），否则退回可学习常量。
        # 语言分支用冻结 LLM 的 [CLS] 句向量作为单一查询（替代原来的 learnable query_token），
        # 与空间体素 token 做 cross-attention。数据集按 num_steps 把 llm_tokens 堆叠成 (B,T,L)，
        # 这里先把 B,T 合并成 (B*T, L) 再喂 LLM（LLM 只接受 2D (batch,seq)）。
        lang = batch.get('llm_tokens', None)
        if lang is not None and hasattr(lang, 'to'):
            lang = lang.to(device)
            if lang.dim() == 3:
                b_, t_, l_ = lang.shape
                lang_flat = lang.reshape(b_ * t_, l_)
            else:
                lang_flat = lang
            llm_out = self.llm(input_ids=lang_flat)
            hid = llm_out.last_hidden_state if hasattr(llm_out, 'last_hidden_state') else llm_out
            cls = hid[:, 0, :]                              # (B*T 或 B, hidden) [CLS] 句向量
            q = self.lang_proj(cls)                        # (B*T 或 B, llm_dim)
            q = q.unsqueeze(1)                             # (., 1, llm_dim)
            if q.shape[0] != B * T:                       # 2D 输入：(B,1,.) -> 扩到 (B*T,1,.)
                q = q.unsqueeze(1).expand(B, T, 1, self.llm_dim).reshape(B * T, 1, self.llm_dim)
        else:
            q = self.query_token.expand(B * T, -1, -1)     # (B*T, 1, llm_dim)

        fused, attn_weights = self.cross_attn(q, spatial)    # (B*T, Q, llm_dim), Q=1 或 L
        fused = fused.mean(dim=1)                            # 池化掉 query 维 -> (B*T, llm_dim)
        fused = rearrange(fused, "(b t) d -> b t d", b=B, t=T).mean(dim=1)  # (B, llm_dim)

        action_pred = self.action_head(fused)

        action_loss = torch.tensor(0.0, device=device)
        if 'action' in batch:
            action_target = batch['action']
            if action_target.dim() > 1:
                # 体素数据集按 num_steps 堆叠了 T 个动作，取序列最后一步的"下一步动作"作目标
                action_target = action_target[:, -1].squeeze(-1)
            action_loss = F.cross_entropy(action_pred, action_target.long())

        total_loss = self.loss_weights['pred'] * pred_loss + self.loss_weights['action'] * action_loss

        return {
            'losses': {'pred_loss': pred_loss, 'action_loss': action_loss, 'total_loss': total_loss},
            'action_pred': action_pred,
            'jepa_emb': jepa_emb,
            'emb': jepa_emb,
            'attn_weights': attn_weights,
        }
