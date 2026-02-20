import torch
import torch.nn.functional as F
import torch.nn as nn

import math

class PGDP(nn.Module):
    def __init__(self, feat_channels, out_channels, sigma_channels=1):
        super().__init__()
        
        in_channels = feat_channels + sigma_channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=1, padding=0, bias=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=1, padding=0, bias=True)
        )

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=1, padding=0, bias=True)
        )

        self.proj2a = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.proj2b = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.proj1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

        self.proj_f2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.proj_f1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        

    def forward(self, stage2, stage1, stage0, sigma):
        '''
        - Stage2, 1, 0과 sigma 전달받기
        - sigma는 stage feature size에 맞춰 downsampling
        - 각 stage와 sigma를 합친 후 stage의 scale별 2d gaussian pr 예측
        -> 2d feature map에 대한 2d gaussian dist map
        y2 = x * (1 + M_pd)
        '''

        sigma2 = F.interpolate(sigma, size=stage2.shape[-2:])
        sigma1 = F.interpolate(sigma, size=stage1.shape[-2:])
        sigma0 = F.interpolate(sigma, size=stage0.shape[-2:])

        in2 = torch.cat([stage2, sigma2], dim=1)
        in1 = torch.cat([stage1, sigma1], dim=1)
        in0 = torch.cat([stage0, sigma0], dim=1)

        '''stage2'''
        conv2_raw = self.conv2(in2)
        # M_pd2
        p2 = torch.sigmoid(conv2_raw)

        # stage1과 합칠 feature
        conv2_up_s1 = F.interpolate(conv2_raw, size=stage1.shape[-2:])
        conv2_ref1 = self.proj2a(conv2_up_s1)
        # stage0과 합칠 feature
        conv2_up_s0 = F.interpolate(conv2_ref1, size=stage0.shape[-2:])
        conv2_ref0 = self.proj2b(conv2_up_s0) # ref : refined
        

        '''stage1'''
        conv1_raw = self.conv1(in1)
        # stage0과 합칠 feature
        conv1_up_s0 = F.interpolate(conv1_raw, size=stage0.shape[-2:])
        conv1_ref0 = self.proj1(conv1_up_s0)

        # skip connection
        fused2_1 = conv2_up_s1 + conv1_raw
        fused2_1 = self.proj_f2(fused2_1)

        # M_pd1
        p1 = torch.sigmoid(fused2_1)

        '''stage0'''
        conv0_raw = self.conv0(in0)

        # skip connection
        fused_all = conv2_ref0 + conv1_ref0 + conv0_raw
        fused_all = self.proj_f1(fused_all)
        # M_pd0
        p0 = torch.sigmoid(fused_all)

        y2 = stage0 * (1 + p0)

        return p2, p1, p0, y2

        
