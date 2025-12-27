import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 기존
class Read_ignore(nn.Module):
    def __init__(self, start_index=1):
        super(Read_ignore, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


# ======READ MODULE들은 CLS TOKEN 처리용 모듈======
# swin용
# class Read_ignore(nn.Module):
#     def __init__(self, start_index=0):
#         super(Read_ignore, self).__init__()
#         self.start_index = start_index

#     def forward(self, x):
#         # Swin 같이 CLS 토큰이 없는 백본에서도 안전하게 쓰기 위해
#         # 아무 토큰도 버리지 않고 그대로 반환
#         return x


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

# 기존
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


# swin용
# class Resample(nn.Module):
#     def __init__(self, p, s, image_height, emb_dim, resample_dim):
#         super(Resample, self).__init__()
#         '''
#         1. 크기와 채널을 stage별로 통일 
#         2. conv1x1로 채널 통일 후 해상도 up/down sampling
#         3. stage별 s(배수)값에 맞춤
#         '''
#         # conv1은 첫 입력의 채널 수를 보고 나중에 만들기 위해 None으로 시작
#         self.conv1 = None
#         self.resample_dim = resample_dim
#         self.s = s

#         # s 값에 따라 해상도 조정용 conv2 설정
#         if s == 4:
#             # 1/4 해상도 -> x4 업샘플
#             self.conv2 = nn.ConvTranspose2d(
#                 resample_dim, resample_dim,
#                 kernel_size=4, stride=4, padding=0, bias=True
#             )
#         elif s == 8:
#             # 1/8 해상도 -> x2 업샘플
#             self.conv2 = nn.ConvTranspose2d(
#                 resample_dim, resample_dim,
#                 kernel_size=2, stride=2, padding=0, bias=True
#             )
#         elif s == 16:
#             # 1/16 해상도 그대로 사용
#             self.conv2 = nn.Identity()
#         else:
#             # 1/32 해상도 -> x0.5 (다운샘플)
#             self.conv2 = nn.Conv2d(
#                 resample_dim, resample_dim,
#                 kernel_size=2, stride=2, padding=0, bias=True
#             )

#     def forward(self, x):
#         # x: [B, C_in, H, W]

#         # 첫 forward에 들어온 채널 수를 보고 conv1 생성
#         if self.conv1 is None:
#             in_channels = x.shape[1]
#             self.conv1 = nn.Conv2d(
#                 in_channels,
#                 self.resample_dim,
#                 kernel_size=1, stride=1, padding=0, bias=True
#             ).to(x.device)

#         x = self.conv1(x)  # [B, resample_dim, H, W]
#         x = self.conv2(x)  # 업/다운샘플
#         return x


class Reassemble(nn.Module):
    def __init__(self, image_size, read, p, s, emb_dim, resample_dim):
        """
        p = patch size
        s = coefficient resample
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}

        - 두 모달리티의 (CAM LIDAR) FEATURE MAP을 항상 [B, C, H, W]로 재구성 목적
        ViT   : [B, N ,C]      ->      [B, C, H, W]
        Swin  : [B, H, W, C]   ->      [B, C, H, W]

        - 또한 Cam과 lidar간 feature 정규화를 한다
        Cam   : 96x96
        Lidar : 64x256
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

    # 기존
    def forward(self, x):
        x = self.read(x)
        x = self.concat(x)
        x = self.resample(x)
        return x

    # swin용
    # def forward(self, x):
    #     """
    #     x:
    #       - Swin 계열: [B, H, W, C]  (hook에서 그대로 들어옴)
    #       - ViT 계열: [B, N, C]     (토큰 시퀀스)
    #     """

    #     if x.dim() == 4:
    #         # Swin: [B, H, W, C] -> [B, C, H, W]
    #         b, h, w, c = x.shape
    #         x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

    #         # Swin에는 CLS 토큰이 없고, 이미 2D map이므로 read/concat 생략
    #         # 바로 Resample로 보냄

    #     else:
    #         # ViT: [B, N, C]
    #         x = self.read(x)   # Read_ignore / Read_add / Read_projection
    #         x = self.concat(x) # [B, C, H, W]

    #     # 공통: NCHW 상태에서 Resample
    #     x = self.resample(x)   # [B, resample_dim, H_out, W_out]
    #     return x


