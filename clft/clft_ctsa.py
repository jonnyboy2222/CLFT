import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTSA(nn.Module):
    """
    Class Token Self-Attention (CTSA)
    Input:  x  shape (B, T, D)   where T = 2K (cam K + lidar K), D = emb_dim
    Output: y  shape (B, T, D)
    """
    def __init__(self, emb_dim: int = 768, num_heads: int = 12,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 use_bias: bool = True, use_ln: bool = True):
        super().__init__()

        assert emb_dim % num_heads == 0, \
            f"emb_dim({emb_dim}) must be divisible by num_heads({num_heads})."

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Standard: one linear that produces qkv together (more efficient)
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=use_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(emb_dim, emb_dim, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_ln = use_ln
        self.ln = nn.LayerNorm(emb_dim) if use_ln else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        assert x.dim() == 3, f"Expected x to be (B,T,D), got {tuple(x.shape)}"
        B, T, D = x.shape
        assert D == self.emb_dim, f"Expected last dim {self.emb_dim}, got {D}"

        # Pre-LN for stability (recommended)
        x_in = x
        x = self.ln(x)

        # qkv: (B, T, 3D) -> (B, T, 3, H, Dh) -> (3, B, H, T, Dh)
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, T, Dh)

        # scores: (B, H, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)

        # out: (B, H, T, Dh) -> (B, T, H, Dh) -> (B, T, D)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)

        out = self.proj(out)
        out = self.proj_drop(out)

        # Residual
        return x_in + out


# if __name__ == "__main__":
#     # check
#     B, K, D = 2, 3, 768
#     T = 2 * K
#     x = torch.randn(B, T, D)

#     m = CTSA(emb_dim=D, num_heads=32, attn_drop=0.0, proj_drop=0.0, use_ln=True)
#     y = m(x)
#     print("in:", x.shape, "out:", y.shape)
