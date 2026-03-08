import torch
import torch.nn as nn
import torch.nn.functional as F


class CWT(nn.Module):
    """Attention-based Class-wise Tokenizer (CWT)

    목적: ViT patch tokens에서 K개의 class-wise anchor token을 attention으로 추출.

    Input
      x        : (B, N, D) ViT tokens. CLS가 포함되어 있으면 x[:,0]이 CLS.
      mask_tok : (B, N_patch) 또는 (B, N_patch, 1) patch-token mask (1=valid, 0=invalid).
                 *인자를 넘겨준 경우에만* masking을 적용.
      has_cls_token: x에 CLS가 포함되어 있으면 True.

    Output
      class_tokens : (B, K, D)

    설계
      - K개의 learnable class queries (slot)이 Q.
      - patch tokens가 K/V.
      - cross-attention 결과를 residual로 query에 더해 anchor를 안정화:
          class_tokens = Q + Attn(Q, K/V)

    NOTE
      - mask_tok은 CLS를 제외한 patch token과 1:1 대응해야 함.
        (즉, x가 CLS를 포함하면 mask_tok은 x[:,1:,:]의 길이 N_patch와 동일)
    """

    def __init__(
        self,
        emb_dim: int = 768,
        num_classes: int = 2,
        num_heads: int = 12,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Learnable class-slot queries (K, D)
        self.class_queries = nn.Parameter(torch.randn(num_classes, emb_dim) * init_std)

        # Pre-LN (stability)
        self.q_norm = nn.LayerNorm(emb_dim)
        self.kv_norm = nn.LayerNorm(emb_dim)

        # Projections
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask_tok: torch.Tensor = None, has_cls_token: bool = True):
        # x: (B, N, D)
        if has_cls_token:
            patch = x[:, 1:, :]  # (B, Np, D)
        else:
            patch = x

        B, Np, D = patch.shape

        # Build Q from learnable slots
        q0 = self.class_queries.unsqueeze(0).expand(B, self.num_classes, D)  # (B, K, D)

        # Pre-LN
        qn = self.q_norm(q0)
        kvn = self.kv_norm(patch)

        # Linear projections
        q = self.q_proj(qn)   # (B, K, D)
        k = self.k_proj(kvn)  # (B, Np, D)
        v = self.v_proj(kvn)  # (B, Np, D)

        # Reshape to multi-head
        # q: (B, H, K, Hd), k/v: (B, H, Np, Hd)
        q = q.view(B, self.num_classes, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Np, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Np, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention logits: (B, H, K, Np)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Optional key masking (only if mask_tok is provided)
        if mask_tok is not None:
            m = mask_tok
            # allow (B, Np, 1)
            if m.dim() == 3:
                m = m.squeeze(-1)
            # m: (B, Np) -> (B, 1, 1, Np)
            m = m.unsqueeze(1).unsqueeze(2)
            # mask invalid keys
            attn = attn.masked_fill(m == 0, -1e4)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum: (B, H, K, Hd)
        out = attn @ v

        # Merge heads: (B, K, D)
        out = out.transpose(1, 2).contiguous().view(B, self.num_classes, D)

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)

        # Residual anchor stabilization (always on)
        return q0 + out
