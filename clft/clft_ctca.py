# clft_ctca.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCA(nn.Module):
    """
    Class-Token Cross Attention (Multi-Head Cross-Attn)

    Q: LiDAR patch tokens      (B, Nq, D)
    K: Class tokens (CTSA out) (B, Nk, D)
    V: Class tokens (CTSA out) (B, Nk, D)

    Output:
      delta: (B, Nq, D)  # semantic proposal (Δ)
      (optional) attn: (B, H, Nq, Nk)
    """
    def __init__(
        self,
        emb_dim: int = 768,
        num_heads: int = 32,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        return_attn: bool = False,
    ):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.return_attn = return_attn

        # Pre-LN (recommended for stability)
        self.norm_q = nn.LayerNorm(emb_dim)
        self.norm_kv = nn.LayerNorm(emb_dim)

        # Projections
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=True)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=True)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor):
        """
        Args:
          q_tokens:  (B, Nq, D)  LiDAR patch tokens (CLS excluded)
          kv_tokens: (B, Nk, D)  class tokens from CTSA (cam+xyz), used as K and V

        Returns:
          delta: (B, Nq, D)
          attn (optional): (B, H, Nq, Nk)
        """
        B, Nq, D = q_tokens.shape
        B2, Nk, D2 = kv_tokens.shape
        assert B == B2 and D == D2, "Batch/Dim mismatch between q_tokens and kv_tokens"

        # Pre-LN
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)

        # Linear projections
        q = self.q_proj(q)   # (B, Nq, D)
        k = self.k_proj(kv)  # (B, Nk, D)
        v = self.v_proj(kv)  # (B, Nk, D)

        # Reshape to heads
        # (B, H, N, Dh)
        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nq, Dh)
        k = k.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nk, Dh)
        v = v.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nk, Dh)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum
        out = attn @ v  # (B, H, Nq, Dh)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)  # (B, Nq, D)

        # Output projection
        out = self.out_proj(out)
        out = self.proj_drop(out)

        # Residual: IMPORTANT
        # CTCA output is a "proposal", but we still return it as delta;
        # the actual update L' = L + (m*g)*delta should happen outside.
        delta = out  # (B, Nq, D)

        if self.return_attn:
            return delta, attn
        return delta
