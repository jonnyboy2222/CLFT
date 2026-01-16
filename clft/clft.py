import numpy as np
import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from clft.reassemble import Reassemble
from clft.fusion import Fusion
from clft.head import HeadDepth, HeadSeg

from clft.clft_cwt import CWT
from clft.clft_ctsa import CTSA
from clft.clft_ctca import CTCA
from clft.clft_mask import patch_mask_from_lidar

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

        # cls token 관련 클래스 객체 생성
        self.cwt = CWT()
        self.ctsa = CTSA(attn_drop=0.0, proj_drop=0.0)
        self.ctca = CTCA(attn_drop=0.0, proj_drop=0.0)

        # g = sigmoid(1x1conv([L, Δ, M]))  -> 토큰별 Linear
        # 입력 채널: [L (D), Δ (D), M (1)] => 2D+1
        self.gate_proj = nn.Linear(2 * emb_dim + 1, 1, bias=True)

        # hooks for CWT concat (rgb, xyz 추가)
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

        mask_tok = None
        if modal == "cross_fusion":
            # lidar: (B,C,H,W) projected lidar image
            # patch_size는 __init__ 인자로 들어온 그 patch_size를 그대로 사용
            mask_tok = patch_mask_from_lidar(lidar, patch_size=16, eps=0.0, dtype=torch.float32)  # (B,N,1)


        previous_stage = None

        for i in np.arange(len(self.fusions) - 1, -1, -1):
            hook_to_take = 't' + str(self.hooks[i])

            if modal == 'cross_fusion':
                # 각 모달리티 patch embedding
                activation_result_rgb = self.activation_rgb[hook_to_take]
                activation_result_xyz = self.activation_xyz[hook_to_take]

                if i == 2:
                    cwt_cam2 = self.cwt(activation_result_rgb)
                    cwt_xyz2 = self.cwt(activation_result_xyz)
                    cls_token_stage2 = torch.cat([cwt_cam2, cwt_xyz2], dim=1)   # (B, 2K, D) 같은 형태
                    ctsa_out2 = self.ctsa(cls_token_stage2)                     # (B, Nk, D)  Nk=2K

                    # ----- CTCA + gated update (LiDAR token update) -----
                    cls_xyz2 = activation_result_xyz[:, :1, :]                  # (B,1,D)  CLS 고정
                    L2 = activation_result_xyz[:, 1:, :]                        # (B,N,D)

                    delta2 = self.ctca(L2, ctsa_out2)                           # (B,N,D)  Δ

                    # g = sigmoid(Linear([L, Δ, M]))
                    if mask_tok is None:
                        raise RuntimeError("mask_tok is None in cross_fusion; check mask creation.")
                    gate_in2 = torch.cat([L2, delta2, mask_tok], dim=-1)        # (B,N,2D+1)
                    g2 = torch.sigmoid(self.gate_proj(gate_in2))                # (B,N,1)

                    L2p = L2 + (g2 * mask_tok) * delta2                         # (B,N,D)
                    activation_result_xyz_updated = torch.cat([cls_xyz2, L2p], dim=1)  # (B,1+N,D)

                    # assemble할 때 업데이트된 xyz 토큰 사용
                    reassemble_result_RGB = self.reassembles_RGB[i](activation_result_rgb)
                    reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result_xyz_updated)

                elif i == 0:
                    cwt_cam0 = self.cwt(activation_result_rgb)
                    cwt_xyz0 = self.cwt(activation_result_xyz)
                    cls_token_stage0 = torch.cat([cwt_cam0, cwt_xyz0], dim=1)
                    ctsa_out0 = self.ctsa(cls_token_stage0)

                    cls_xyz0 = activation_result_xyz[:, :1, :]
                    L0 = activation_result_xyz[:, 1:, :]

                    delta0 = self.ctca(L0, ctsa_out0)

                    gate_in0 = torch.cat([L0, delta0, mask_tok], dim=-1)
                    g0 = torch.sigmoid(self.gate_proj(gate_in0))

                    L0p = L0 + (g0 * mask_tok) * delta0
                    activation_result_xyz_updated = torch.cat([cls_xyz0, L0p], dim=1)

                    reassemble_result_RGB = self.reassembles_RGB[i](activation_result_rgb)
                    reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result_xyz_updated)

                else:
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

        out_depth = None
        out_segmentation = None
        if self.head_depth is not None:
            out_depth = self.head_depth(previous_stage)
        if self.head_segmentation is not None:
            out_segmentation = self.head_segmentation(previous_stage)

        return out_depth, out_segmentation

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