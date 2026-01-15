import torch
import torch.nn as nn
import torch.nn.functional as F


class CWT(nn.Module):
    def __init__(self, emb_dim=768, num_classes=3, hidden=256):
        super(CWT, self).__init__()

        self.simple_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes)
        )
         
    def forward(self, x, has_cls_token=True):
        # x: (B, N, D)
        x_token = x[:, 1:, :] if has_cls_token else x  # (B, N, D)

        s = self.simple_mlp(x_token)          # (B, N, K) k는 class score
        a = F.softmax(s, dim=1)               # (B, N, K) k는 class index
        # dim=1로 클래스별로 n번째 토큰 상대확률

        cls_token = torch.bmm(a.transpose(1, 2), x_token)  # (B, K, D)
        return cls_token # , a


# check
# if __name__ == "__main__":
#     B, N, D = 2, 197, 768  # vit백본: 1 CLS + 196 patches
#     x = torch.randn(B, N, D)

#     cwt = CWT()
#     t, a = cwt(x)

#     print("x shape:", x.shape)

#     print("T:", t.shape)  # (B, 3, 768)
#     print("A:", a.shape)  # (B, N-1, 3)
#     print("sum over tokens:", a.sum(dim=1)[0])  # 각 합이 1이 되어야 함