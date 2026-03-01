import torch
import torch.nn.functional as F
import torch.nn as nn

import math

class PGDP(nn.Module):
    def __init__(self, feat_channels, out_channels, sigma_channels=1):
        super().__init__()

        self.feat_channels = feat_channels
        self.sigma_channels = sigma_channels

        # --- sigma를 feature 채널로 맞춰서 add 하기 위한 projection ---
        # (sigma_channels=1이면 1x1 conv로 C채널로 확장)
        self.sigma_proj = nn.Conv2d(sigma_channels, feat_channels, kernel_size=1, padding=0, bias=True)

        # --- conv 입력은 feat_channels 그대로 ---
        in_channels = feat_channels
        hidden = max(in_channels // 2, 1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, padding=0, bias=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, padding=0, bias=True)
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, padding=0, bias=True)
        )

        # --- learnable upsampling (deconv) ---
        # stage2 -> stage1 : x2
        self.up2_to_1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True)
        # stage1 -> stage0 : x2
        self.up1_to_0 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True)
        # stage2(ref1) -> stage0 : x2
        self.up2ref1_to_0 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True)

        self.proj2a = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.proj2b = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.proj1  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

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

        # --- sigma를 각 stage 해상도로 맞추고, 채널을 feat_channels로 projection ---
        sigma2 = F.interpolate(sigma, size=stage2.shape[-2:], mode="bilinear", align_corners=False)
        sigma1 = F.interpolate(sigma, size=stage1.shape[-2:], mode="bilinear", align_corners=False)
        sigma0 = F.interpolate(sigma, size=stage0.shape[-2:], mode="bilinear", align_corners=False)

        sigma2 = self.sigma_proj(sigma2)
        sigma1 = self.sigma_proj(sigma1)
        sigma0 = self.sigma_proj(sigma0)

        in2 = stage2 + sigma2
        in1 = stage1 + sigma1
        in0 = stage0 + sigma0

        '''stage2'''
        conv2_raw = self.conv2(in2)
        # M_pd2
        p2 = torch.sigmoid(conv2_raw)

        # stage1과 합칠 feature
        conv2_up_s1 = self.up2_to_1(conv2_raw)
        conv2_ref1 = self.proj2a(conv2_up_s1)
        # stage0과 합칠 feature
        conv2_up_s0 = self.up2ref1_to_0(conv2_ref1)
        conv2_ref0 = self.proj2b(conv2_up_s0) # ref : refined
        

        '''stage1'''
        conv1_raw = self.conv1(in1)
        # stage0과 합칠 feature
        conv1_up_s0 = self.up1_to_0(conv1_raw)
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
        p0_fg = p0.max(dim=1, keepdim=True).values   # (B,1,H0,W0) : apply용

        y2 = stage0 * (1 + p0_fg) # 학습은 채널별(클래스별), 증폭은 합쳐서 수행

        with torch.no_grad():
            fg = p0_fg
            fg_flat = fg.flatten(1)
            self.diag = {
                # "pgdp_loss": float(loss.detach().cpu()),
                "p0_fg_mean": float(fg.mean().detach().cpu()),
                "p0_fg_min": float(fg.min().detach().cpu()),
                "p0_fg_p95": float(fg_flat.quantile(0.95, dim=1).mean().detach().cpu()),
                "p0_fg_max": float(fg.max().detach().cpu()),
            }

        # viz
        with torch.no_grad():
            self.last_pgdp_p2 = p2.detach()
            self.last_pgdp_p1 = p1.detach()
            self.last_pgdp_p0 = p0.detach()

        return p2, p1, p0, y2

        
