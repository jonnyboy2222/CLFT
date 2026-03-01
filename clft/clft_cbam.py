import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
        )

    def forward(self, x):
        # GAP + GMP
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=pad, bias=False)

    def forward(self, x):
        # channel-wise avg/max -> (B,1,H,W) each
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, sa_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction=reduction)
        self.sa = SpatialAttention(k=sa_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x