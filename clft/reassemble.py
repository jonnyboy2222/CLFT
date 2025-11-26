import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Read_ignore(nn.Module):
    def __init__(self, start_index=1):
        super(Read_ignore, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class Read_add(nn.Module):
    def __init__(self, start_index=1):
        super(Read_add, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class Read_projection(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(Read_projection, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)
        return self.project(features)


class MyConvTranspose2d(nn.Module):
    def __init__(self, conv, output_size):
        super(MyConvTranspose2d, self).__init__()
        self.output_size = output_size
        self.conv = conv

    def forward(self, x):
        x = self.conv(x, output_size=self.output_size)
        return x


class Resample(nn.Module):
    def __init__(self, p, s, h, emb_dim, resample_dim):
        super(Resample, self).__init__()
        assert (s in [4, 8, 16, 32]), "s must be in [0.5, 4, 8, 16, 32]"
        self.conv1 = nn.Conv2d(emb_dim, resample_dim, kernel_size=1, stride=1, padding=0)
        if s == 4:
            self.conv2 = nn.ConvTranspose2d(resample_dim,
                                resample_dim,
                                kernel_size=4,
                                stride=4,
                                padding=0,
                                bias=True,
                                dilation=1,
                                groups=1)
        elif s == 8:
            self.conv2 = nn.ConvTranspose2d(resample_dim,
                                resample_dim,
                                kernel_size=2,
                                stride=2,
                                padding=0,
                                bias=True,
                                dilation=1,
                                groups=1)
        elif s == 16:
            self.conv2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(resample_dim, resample_dim, kernel_size=2,stride=2, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Reassemble(nn.Module):
    def __init__(self, image_size, read, p, s, emb_dim, resample_dim):
        """
        p = patch size
        s = coefficient resample
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super(Reassemble, self).__init__()
        channels, image_height, image_width = image_size

        #Read
        self.read = Read_ignore()
        if read == 'add':
            self.read = Read_add()
        elif read == 'projection':
            self.read = Read_projection(emb_dim)

        #Concat after read
        self.concat = Rearrange('b (h w) c -> b c h w',
                                c=emb_dim,
                                h=(image_height // p),
                                w=(image_width // p))

        #Projection + Resample
        self.resample = Resample(p, s, image_height, emb_dim, resample_dim)

    # def forward(self, x):
    #     x = self.read(x)
    #     x = self.concat(x)
    #     x = self.resample(x)
    #     return x

    def forward(self, x):
        """
        x: 
        - 일부 백본(ViT 등)은 [B, N, C] 토큰 시퀀스를 직접 내보내고,
        - Swin처럼 conv-ish 구조인 경우 [B, C, H, W] feature map을 내보내기도 한다.
        여기서는 두 경우를 모두 지원하기 위해:
        1) 4D면 먼저 [B, C, H, W] -> [B, N, C] 로 평탄화
        2) 그 다음 read(CLS 처리 등)를 적용
        3) 다시 2D grid로 reshape 후 Resample
        """

        # 1) 만약 hook에서 4D feature map이 들어왔다면, 토큰 시퀀스로 변환
        if x.dim() == 4:
            # x: [B, C, H, W]
            b, c, h, w = x.shape
            # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
            x = x.view(b, c, h * w).transpose(1, 2).contiguous()

        # 여기까지 오면 x는 항상 [B, N, C] 형태
        x = self.read(x)  # Read_ignore / Read_add / Read_projection

        b, n, c = x.shape
        # N = H' * W' 인 완전 제곱수라고 가정 (토큰이 2D grid에서 flatten된 것)
        h = int(n ** 0.5)
        w = n // h
        assert h * w == n, f"Reassemble: token length {n} is not a perfect square"

        # [B, N, C] -> [B, C, H', W']
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)

        # Projection + up/down-sampling
        x = self.resample(x)
        return x

