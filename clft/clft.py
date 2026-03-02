import numpy as np
import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from clft.reassemble import Reassemble
from clft.fusion import Fusion
from clft.head import HeadDepth, HeadSeg

from clft.clft_pgdp import PGDP
from clft.clft_pfim import PFIM
from clft.clft_cbam import CBAM

torch.manual_seed(0)


class CLFT(nn.Module):
    def __init__(self,
                 RGB_tensor_size=None,
                 XYZ_tensor_size=None,
                 patch_size=None,
                 emb_dim=None,
                 resample_dim=None,
                 read=None,
                 hooks=None,
                 reassemble_s=None,
                 nclasses=None,
                 type=None,
                 model_timm=None
                 # num_layers_encoder=24,
                 # transformer_dropout=0,
                 ):
        """
        Focus on Depth
        type : {"full", "depth", "segmentation"}
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()

        self.transformer_encoders = timm.create_model(model_timm, pretrained=True)
        self.type_ = type

        # hooks for rgb, xyz
        self.activation_rgb = {}
        self.activation_xyz = {}
        self.activation = {}

        # vit encoder에서 빼올 때 같은 값을 저장하지 않기 위한 조건분기용 변수
        # single, rgb, xyz
        self._hook_target = "single"

        # Register hooks
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        # Reassembles Fusion
        self.reassembles_RGB = []
        self.reassembles_XYZ = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles_RGB.append(Reassemble(RGB_tensor_size, read, patch_size, s, emb_dim, resample_dim))
            self.reassembles_XYZ.append(Reassemble(XYZ_tensor_size, read, patch_size, s, emb_dim, resample_dim))
            self.fusions.append(Fusion(resample_dim))
        self.reassembles_RGB = nn.ModuleList(self.reassembles_RGB)
        self.reassembles_XYZ = nn.ModuleList(self.reassembles_XYZ)
        self.fusions = nn.ModuleList(self.fusions)

        # P&P module
        self.pfim = PFIM(
            in_channels=resample_dim,
            hidden_channels=resample_dim // 2
        )
        self.pgdp = PGDP(
            feat_channels=resample_dim,
            out_channels=2
        )
        self.cbam1_rgb = CBAM(resample_dim)
        self.cbam2_rgb = CBAM(resample_dim)

        self.cbam1_xyz = CBAM(resample_dim)
        self.cbam2_xyz = CBAM(resample_dim)

        self.rgb_reassemble = {}
        self.xyz_reassemble = {}

        #Head
        if type == "full":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
        elif type == "depth":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = None
        else:
            self.head_depth = None
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)

    def forward(self, rgb, lidar, modal='rgb'):
        # key clear
        # forward할 때 이전 값이 남아있을 경우를 대비
        self.activation.clear()
        self.activation_rgb.clear()
        self.activation_xyz.clear()
        extras_rgb = {}
        extras_xyz = {}

        if modal == 'cross_fusion':
            self._hook_target = "rgb"
            _ = self.transformer_encoders(rgb)

            self._hook_target = "xyz"
            _ = self.transformer_encoders(lidar)

            self._hook_target = "single"  # reset
        elif modal == 'rgb':
            self._hook_target = "single"
            _ = self.transformer_encoders(rgb)
        elif modal == 'lidar':
            self._hook_target = "single"
            _ = self.transformer_encoders(lidar)
        else:
            raise ValueError(f"Unknown modal='{modal}'. Expected 'rgb', 'lidar', or 'cross_fusion'.")
        
        # modal은 cross fusion 기본
        for i in range(4):
            hook_to_take = 't' + str(self.hooks[i])

            # 각 모달리티 patch embedding
            activation_result_rgb = self.activation_rgb[hook_to_take]
            activation_result_xyz = self.activation_xyz[hook_to_take]

            reassemble_result_RGB = self.reassembles_RGB[i](activation_result_rgb)
            reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result_xyz)

            self.rgb_reassemble[i] = reassemble_result_RGB
            self.xyz_reassemble[i] = reassemble_result_XYZ

            if i == 0:
                raw_stage0_rgb = reassemble_result_RGB
                raw_stage0_xyz = reassemble_result_XYZ


        # pfim 실행
        y1_rgb, pfim_loss_rgb, info_map_rgb = self.pfim(self.rgb_reassemble[0])
        y1_xyz, pfim_loss_xyz, info_map_xyz = self.pfim(self.xyz_reassemble[0])

        # pgdp 실행
        p2_rgb, p1_rgb, p0_rgb, y2_rgb = self.pgdp(
            self.rgb_reassemble[2],
            self.rgb_reassemble[1],
            self.rgb_reassemble[0],
            info_map_rgb
        )

        p2_xyz, p1_xyz, p0_xyz, y2_xyz = self.pgdp(
            self.xyz_reassemble[2],
            self.xyz_reassemble[1],
            self.xyz_reassemble[0],
            info_map_xyz
        )
        
        a1_rgb = self.cbam1_rgb(y1_rgb)
        a2_rgb = self.cbam2_rgb(y2_rgb)

        a1_xyz = self.cbam1_xyz(y1_xyz)
        a2_xyz = self.cbam2_xyz(y2_xyz)

        updated_s0_rgb = a1_rgb + a2_rgb
        updated_s0_xyz = a1_xyz + a2_xyz

        self.rgb_reassemble[0] = updated_s0_rgb
        self.xyz_reassemble[0] = updated_s0_xyz

        previous_stage = None

        for i in np.arange(len(self.fusions) - 1, -1, -1):
            fusion_result = self.fusions[i](self.rgb_reassemble[i], self.xyz_reassemble[i], previous_stage, modal)
            previous_stage = fusion_result
            

        # ===== P&P Diagnostics =====
        with torch.no_grad():
            eps = 1e-6

            # rgb
            # info map stats
            info_mean_rgb = info_map_rgb.mean()
            info_p95_rgb  = torch.quantile(info_map_rgb.flatten(), 0.95)

            # PGDP foreground prior
            p0_fg_rgb = p0_rgb.max(dim=1, keepdim=True).values
            p0_mean_rgb = p0_fg_rgb.mean()
            p0_p95_rgb  = torch.quantile(p0_fg_rgb.flatten(), 0.95)

            # CBAM update strength
            delta_s0_rgb = updated_s0_rgb - raw_stage0_rgb
            delta_ratio_rgb = delta_s0_rgb.norm() / (raw_stage0_rgb.norm() + eps)


            # xyz
            # info map stats
            info_mean_xyz = info_map_xyz.mean()
            info_p95_xyz  = torch.quantile(info_map_xyz.flatten(), 0.95)

            # PGDP foreground prior
            p0_fg_xyz = p0_xyz.max(dim=1, keepdim=True).values
            p0_mean_xyz = p0_fg_xyz.mean()
            p0_p95_xyz  = torch.quantile(p0_fg_xyz.flatten(), 0.95)

            # CBAM update strength
            delta_s0_xyz = updated_s0_xyz - raw_stage0_xyz
            delta_ratio_xyz = delta_s0_xyz.norm() / (raw_stage0_xyz.norm() + eps)

        extras_rgb = {
            "pfim_loss_rgb": pfim_loss_rgb,
            "p2_rgb": p2_rgb, 
            "p1_rgb": p1_rgb, 
            "p0_rgb": p0_rgb,
            "info_mean_rgb": info_mean_rgb.detach(),
            "info_p95_rgb": info_p95_rgb.detach(),
            "p0_mean_rgb": p0_mean_rgb.detach(),
            "p0_p95_rgb": p0_p95_rgb.detach(),
            "delta_ratio_rgb": delta_ratio_rgb.detach(),
        }

        extras_xyz = {
            "pfim_loss_xyz": pfim_loss_xyz,
            "p2_xyz": p2_xyz, 
            "p1_xyz": p1_xyz, 
            "p0_xyz": p0_xyz,
            "info_mean_xyz": info_mean_xyz.detach(),
            "info_p95_xyz": info_p95_xyz.detach(),
            "p0_mean_xyz": p0_mean_xyz.detach(),
            "p0_p95_xyz": p0_p95_xyz.detach(),
            "delta_ratio_xyz": delta_ratio_xyz.detach(),
        }

        out_depth = None
        out_segmentation = None
        if self.head_depth is not None:
            out_depth = self.head_depth(previous_stage)
        if self.head_segmentation is not None:
            out_segmentation = self.head_segmentation(previous_stage)

        return out_depth, out_segmentation, extras_rgb, extras_xyz

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                # 현재 hook target에 맞추어 output 저장
                # model/input/output은 파이토치에서 필요한 인자
                if self._hook_target == "rgb":
                    self.activation_rgb[name] = output
                elif self._hook_target == "xyz":
                    self.activation_xyz[name] = output
                else:
                    self.activation[name] = output
            return hook

        for h in hooks:
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t' + str(h)))