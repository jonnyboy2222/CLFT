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
        self.cbam1 = CBAM(resample_dim)
        self.cbam2 = CBAM(resample_dim)

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
        extras = {}

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


        previous_stage = None

        for i in np.arange(len(self.fusions) - 1, -1, -1):
            hook_to_take = 't' + str(self.hooks[i])

            if modal == 'cross_fusion':
                # 각 모달리티 patch embedding
                activation_result_rgb = self.activation_rgb[hook_to_take]
                activation_result_xyz = self.activation_xyz[hook_to_take]

                reassemble_result_RGB = self.reassembles_RGB[i](activation_result_rgb)
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result_xyz)

            elif modal == 'rgb':
                activation_result = self.activation[hook_to_take]
                reassemble_result_RGB = self.reassembles_RGB[i](activation_result)
                reassemble_result_XYZ = torch.zeros_like(reassemble_result_RGB)

            elif modal == 'lidar':
                activation_result = self.activation[hook_to_take]
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result)
                reassemble_result_RGB = torch.zeros_like(reassemble_result_XYZ)

            fusion_result = self.fusions[i](reassemble_result_RGB, reassemble_result_XYZ, previous_stage, modal)
            previous_stage = fusion_result

            # i마다 stage fusion feature 보존
            if i == 2:
                stage2_feat = fusion_result
            elif i == 1:
                stage1_feat = fusion_result
            elif i == 0:
                stage0_feat = fusion_result

        
        # pfim 실행
        y1, pfim_loss, info_map = self.pfim(stage0_feat)

        # pgdp 실행
        p2, p1, p0, y2 = self.pgdp(
            stage2_feat,
            stage1_feat,
            stage0_feat,
            info_map
        )

        # 증폭분만 활용
        pfim_delta = y1 - stage0_feat
        pgdp_delta = y2 - stage0_feat
        
        a1 = self.cbam1(pfim_delta)
        a2 = self.cbam2(pgdp_delta)


        updated_s0 = stage0_feat + a1 + a2
        previous_stage = updated_s0

        # ===== P&P Diagnostics =====
        with torch.no_grad():
            eps = 1e-6

            # info map stats
            info_mean = info_map.mean()
            info_std  = info_map.std()
            info_p95  = torch.quantile(info_map.flatten(), 0.95)

            # PGDP foreground prior
            p0_fg = p0.max(dim=1, keepdim=True).values
            p0_mean = p0_fg.mean()
            p0_p95  = torch.quantile(p0_fg.flatten(), 0.95)

            # CBAM update strength
            delta_s0 = updated_s0 - stage0_feat
            delta_ratio = delta_s0.norm() / (stage0_feat.norm() + eps)

            y1_ratio = y1.norm() / (updated_s0.norm() + eps)
            y2_ratio = y2.norm() / (updated_s0.norm() + eps)

        extras = {
            "pfim_loss": pfim_loss,
            "p2": p2, 
            "p1": p1, 
            "p0": p0,
            "info_mean": info_mean.detach(),
            "info_std": info_std.detach(),
            "info_p95": info_p95.detach(),
            "p0_mean": p0_mean.detach(),
            "p0_p95": p0_p95.detach(),
            "delta_ratio": delta_ratio.detach(),
            "y1_ratio": y1_ratio.detach(),
            "y2_ratio": y2_ratio.detach(),
        }

        out_depth = None
        out_segmentation = None
        if self.head_depth is not None:
            out_depth = self.head_depth(previous_stage)
        if self.head_segmentation is not None:
            out_segmentation = self.head_segmentation(previous_stage)

        return out_depth, out_segmentation, extras

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