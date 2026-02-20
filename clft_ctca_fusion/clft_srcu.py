import torch
import torch.nn as nn
import torch.nn.functional as F

def nchw_to_nhwc(x):  # (B,C,H,W) -> (B,H,W,C)
    return x.permute(0, 2, 3, 1).contiguous()

def nhwc_to_nchw(x):  # (B,H,W,C) -> (B,C,H,W)
    return x.permute(0, 3, 1, 2).contiguous()

class WindowPartition(nn.Module):
    def __init__(self, window_size: int):
        super().__init__()
        self.ws = window_size

    def partition(self, x):  # x: (B,H,W,C)
        B, H, W, C = x.shape
        ws = self.ws
        # assume H%ws==0, W%ws==0 for draft
        x = x.view(B, H//ws, ws, W//ws, ws, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H/ws, W/ws, ws, ws, C)
        windows = windows.view(-1, ws*ws, C)  # (B*nW, N, C)
        return windows, (H, W, B)

    def reverse(self, windows, meta):
        H, W, B = meta
        ws = self.ws
        C = windows.shape[-1]
        x = windows.view(B, H//ws, W//ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H/ws, ws, W/ws, ws, C)
        x = x.view(B, H, W, C)
        return x

class WMSA(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # relative position bias
        # (2ws-1)*(2ws-1) positions
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)*(2*window_size-1), num_heads)
        )

        # precompute relative_position_index for N x N
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))  # (2, ws, ws)
        coords_flat = coords.flatten(1)  # (2, N)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()  # (N, N, 2)
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= (2*window_size - 1)
        relative_position_index = rel.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, attn_mask=None):
        # x: (B*nW, N, C)
        BnW, N, C = x.shape
        qkv = self.qkv(x)  # (BnW, N, 3C)
        qkv = qkv.view(BnW, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, BnW, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (BnW, heads, N, N)

        # add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()  # (heads, N, N)
        attn = attn + bias.unsqueeze(0)  # (BnW, heads, N, N)

        if attn_mask is not None:
            # attn_mask: (nW, N, N) or (1, nW, N, N)
            # reshape attn to (B, nW, heads, N, N)
            nW = attn_mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        out = attn @ v  # (BnW, heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(BnW, N, C)
        out = self.proj(out)
        return out

class SWMSA(nn.Module):
    def __init__(self, dim, window_size=7, shift_size=0, num_heads=4):
        super().__init__()
        self.ws = window_size
        self.ss = shift_size
        assert 0 <= shift_size < window_size

        self.partition = WindowPartition(window_size)
        self.attn = WMSA(dim, window_size, num_heads)
        self.register_buffer("attn_mask", None, persistent=False)

    def build_mask(self, H, W, device):
        # only needed for shift
        ws, ss = self.ws, self.ss
        if ss == 0:
            return None

        img_mask = torch.zeros((1, H, W, 1), device=device)  # (1,H,W,1)
        cnt = 0
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # shift mask and window partition
        img_mask = torch.roll(img_mask, shifts=(-ss, -ss), dims=(1, 2))
        mask_windows, _ = self.partition.partition(img_mask)  # (nW, N, 1)
        mask_windows = mask_windows.view(-1, ws*ws)
        attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]  # (nW, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x):  # x: (B,H,W,C)
        B, H, W, C = x.shape
        ws, ss = self.ws, self.ss

        if ss > 0:
            x = torch.roll(x, shifts=(-ss, -ss), dims=(1, 2))

        x_windows, meta = self.partition.partition(x)  # (B*nW, N, C)

        # build / cache mask per (H,W)
        if ss > 0 and (self.attn_mask is None or self.attn_mask.shape[0] != (H//ws)*(W//ws)):
            self.attn_mask = self.build_mask(H, W, x.device)

        x_windows = self.attn(x_windows, attn_mask=self.attn_mask if ss > 0 else None)
        x = self.partition.reverse(x_windows, meta)

        if ss > 0:
            x = torch.roll(x, shifts=(ss, ss), dims=(1, 2))
        return x

class SRCU(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = SWMSA(dim, window_size, shift_size=0, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = SWMSA(dim, window_size, shift_size=window_size//2, num_heads=num_heads)

        hidden = int(dim * mlp_ratio)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):  # x: (B,C,H,W)
        x = nchw_to_nhwc(x)

        # W-MSA
        x = x + self.attn1(self.norm1(x))
        # SW-MSA
        x = x + self.attn2(self.norm2(x))
        # FFN
        x = x + self.mlp(self.norm3(x))

        x = nhwc_to_nchw(x)
        return x
