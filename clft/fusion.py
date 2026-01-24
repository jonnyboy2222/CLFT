import numpy as np
import torch
import torch.nn as nn


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class Fusion(nn.Module):
    def __init__(self, resample_dim, alpha = 0.10):
        super(Fusion, self).__init__()
        #self.res_conv1 = ResidualConvUnit(resample_dim)
        self.res_conv_xyz = ResidualConvUnit(resample_dim)
        self.res_conv_rgb = ResidualConvUnit(resample_dim)

        self.res_conv2 = ResidualConvUnit(resample_dim)
        #self.resample = nn.ConvTranspose2d(resample_dim, resample_dim, kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1)

        # fusion gate
        self.alpha = float(alpha)
        # g = sigmoid( conv1x1([C, L', M]) )
        # C: (B,D,H,W), L': (B,D,H,W), M: (B,1,H,W)  -> in_ch = 2D+1
        self.gate_conv = nn.Conv2d(2 * resample_dim + 1, 1, kernel_size=1, bias=True)

    def forward(self, rgb, lidar, previous_stage=None, modal = 'rgb', mask=None):
        if previous_stage == None:
                previous_stage = torch.zeros_like(rgb)

        if modal == 'rgb':
            output_stage1_rgb = self.res_conv_rgb(rgb)
            output_stage1_lidar = torch.zeros_like(output_stage1_rgb)
        if modal == 'lidar':
            output_stage1_lidar = self.res_conv_xyz(lidar)
            output_stage1_rgb = torch.zeros_like(output_stage1_lidar)
        if modal == 'cross_fusion': 
            # output_stage1_rgb = self.res_conv_rgb(rgb)
            # output_stage1_lidar = self.res_conv_xyz(lidar)

            C  = self.res_conv_rgb(rgb)
            Lp = self.res_conv_xyz(lidar)

            # mask: (B,1,H,W) expected
            if mask is None:
                # fallback: 없으면 전부 1로 (or raise)
                mask = torch.ones((C.size(0), 1, C.size(2), C.size(3)), device=C.device, dtype=C.dtype)
            else:
                mask = mask.to(device=C.device, dtype=C.dtype)

            gate_in = torch.cat([C, Lp, mask], dim=1)      # (B,2D+1,H,W)
            g = torch.sigmoid(self.gate_conv(gate_in))     # (B,1,H,W)

            scale = self.alpha + (1.0 - self.alpha) * g    # (B,1,H,W)
            Fused = C + scale * (mask * Lp)                  


        # output_stage1 = output_stage1_lidar + output_stage1_rgb + previous_stage
        # prev + F
        output_stage1 = Fused + previous_stage
        output_stage2 = self.res_conv2(output_stage1)
        
        output_stage2 = nn.functional.interpolate(output_stage2, scale_factor=2, mode="bilinear", align_corners=True)
        return output_stage2


            
