import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        # 기존, ups v3
        # self.conv1 = nn.Conv2d(
        #     features, features, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv2 = nn.Conv2d(
        #     features, features, kernel_size=3, stride=1, padding=1, bias=True)
        # self.relu = nn.ReLU(inplace=True)

        # ups v1, v2
        # rcu 구조변경
        self.dw1 = nn.Conv2d(features, features, 3, padding=1, groups=features, bias=True)
        self.pw1 = nn.Conv2d(features, features, 1, bias=True)
        self.dw2 = nn.Conv2d(features, features, 3, padding=1, groups=features, bias=True)
        self.pw2 = nn.Conv2d(features, features, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        # 기존
        # out = self.relu(x)
        # out = self.conv1(out)
        # out = self.relu(out)
        # out = self.conv2(out)

        # ups v1, v2
        out = self.dw1(x)
        out = self.pw1(out)
        out = self.relu(out)

        out = self.dw2(out)
        out = self.pw2(out)
        out = self.relu(out)

        # ups v3
        # out = self.conv1(x)
        # out = self.relu(out)
        # out = self.conv2(out)
        # out = self.relu(out)
        
        return out + x


class Fusion(nn.Module):
    def __init__(self, resample_dim):
        super(Fusion, self).__init__()

        # fpn적용
        # lateral 1x1 conv: C_l -> P_l 채널 정렬
        # self.lateral_rgb = nn.Conv2d(resample_dim, resample_dim, kernel_size=1)
        # self.lateral_xyz = nn.Conv2d(resample_dim, resample_dim, kernel_size=1)
        # # top-down projection (optional, 그냥 identity conv)
        # self.topdown_proj = nn.Conv2d(resample_dim, resample_dim, kernel_size=1)
        # # FPN smoothing: P_l을 부드럽게 만드는 3x3 conv (RCU 기존구조)
        # self.smooth = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True)
        # )

        #self.res_conv1 = ResidualConvUnit(resample_dim)
        self.res_conv_xyz = ResidualConvUnit(resample_dim)
        self.res_conv_rgb = ResidualConvUnit(resample_dim)

        # 기존
        self.res_conv2 = ResidualConvUnit(resample_dim)
        # 원래주석
        #self.resample = nn.ConvTranspose2d(resample_dim, resample_dim, kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1)

    # 기존
    def forward(self, rgb, lidar, previous_stage=None, modal = 'rgb'):
        if previous_stage == None:
                previous_stage = torch.zeros_like(rgb)

        if modal == 'rgb':
            output_stage1_rgb = self.res_conv_rgb(rgb)
            output_stage1_lidar = torch.zeros_like(output_stage1_rgb)
        if modal == 'lidar':
            output_stage1_lidar = self.res_conv_xyz(lidar)
            output_stage1_rgb = torch.zeros_like(output_stage1_lidar)
        if modal == 'cross_fusion': 
            output_stage1_rgb = self.res_conv_rgb(rgb)
            output_stage1_lidar = self.res_conv_xyz(lidar)

        output_stage1 = output_stage1_lidar + output_stage1_rgb + previous_stage
        output_stage2 = self.res_conv2(output_stage1)
        
        # allign_corners=False로 변경
        output_stage2 = F.interpolate(output_stage2, scale_factor=2, mode="bilinear", align_corners=False)
        return output_stage2

    # fpn적용
    # def forward(self, rgb, lidar, previous_stage=None, modal='rgb'):
    #     # 1) top-down: 이전 P_{l+1}를 현재 해상도에 맞게 upsample
    #     if previous_stage is None:
    #         td = torch.zeros_like(rgb)
    #     else:
    #         td = F.interpolate(previous_stage, size=rgb.shape[-2:], mode="bilinear", align_corners=True)
    #         td = self.topdown_proj(td)

    #     # 2) lateral: 현재 레벨 C_l -> P_l 채널 정렬 + modality 별 처리
    #     if modal == 'rgb':
    #         lat = self.lateral_rgb(self.res_conv_rgb(rgb))
    #     elif modal == 'lidar':
    #         lat = self.lateral_xyz(self.res_conv_xyz(lidar))
    #     elif modal == 'cross_fusion':
    #         lat_rgb = self.lateral_rgb(self.res_conv_rgb(rgb))
    #         lat_xyz = self.lateral_xyz(self.res_conv_xyz(lidar))
    #         lat = lat_rgb + lat_xyz
    #     else:
    #         raise ValueError(f"Unknown modal: {modal}")

    #     # 3) FPN-style sum + smoothing: P_l = smooth(lat + td)
    #     p_l = self.smooth(lat + td)

    #     # 4) 다음 (더 고해상도) stage에 넘겨줄 top-down feature는 x2 upsample
    #     out = F.interpolate(p_l, scale_factor=2.0, mode="bilinear", align_corners=True)
    #     return out   # 필요하면 p_l도 리턴해서 최종 prediction에 쓰기


            
