import torch
import torch.nn as nn
import torch.nn.functional as F


class CWT(nn.Module):
    """
    Class-wise Tokenizer (CWT)
    Input   : (B, N, D)     ViT patch embedding(including class token itself)
    Output  : (B, K, D)     where K = num_classes
    """
    def __init__(self, emb_dim=768, num_classes=3, hidden=256):
        super(CWT, self).__init__()

        self.simple_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes)
        )
         
    def forward(self, x, mask_tok=None, has_cls_token=True):
        # x: (B, N, D)
        x_token = x[:, 1:, :] if has_cls_token else x  # (B, N, D)
        B, N, D = x_token.shape

        s = self.simple_mlp(x_token)          # (B, N, K) k: class score
        # a = F.softmax(s, dim=1)               # (B, N, K) k: class index
        # # relative class-wise prob of n^th token

        if mask_tok is None:
            # normal token-competition over tokens
            a = F.softmax(s, dim=1)  # (B, Np, K)
        else:
            m = mask_tok
            if m.dim() == 2:
                m = m.unsqueeze(-1)  # (B, Np, 1)
            if m.shape[0] != B or m.shape[1] != N:
                raise ValueError(
                    f"[CWT] mask_tok shape {tuple(m.shape)} must match (B,Np,1)={(B,N,1)}"
                )
            m = m.to(device=s.device, dtype=torch.float32)

            # 1) hard-exclude invalid tokens in logits BEFORE softmax
            #    -> masked tokens get ~0 probability after softmax
            s_masked = s.float().masked_fill(m == 0, -1e4)

            # 2) softmax over tokens (dim=1)
            a = F.softmax(s_masked, dim=1)  # (B, Np, K)

            # 3) enforce exact zero on masked tokens, then renormalize over valid tokens
            a = a * m
            denom = a.sum(dim=1, keepdim=True).clamp_min(1e-6)  # (B,1,K)
            a = a / denom

            a = a.to(dtype=s.dtype)

        cls_token = torch.bmm(a.transpose(1, 2), x_token)  # (B, K, D)
        return cls_token # , a


# check
# if __name__ == "__main__":
#     B, N, D = 2, 197, 768  # ViT: 1 cls token + 196 patches
#     x = torch.randn(B, N, D)

#     cwt = CWT()
#     t, a = cwt(x)

#     print("x shape:", x.shape)

#     print("T:", t.shape)  # (B, 3, 768)
#     print("A:", a.shape)  # (B, N-1, 3)
#     print("sum over tokens:", a.sum(dim=1)[0])  # add up to 1