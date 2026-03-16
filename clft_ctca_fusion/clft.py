import numpy as np
import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from clft_ctca_fusion.reassemble import Reassemble
from clft_ctca_fusion.fusion import Fusion
from clft_ctca_fusion.head import HeadDepth, HeadSeg

from clft_ctca_fusion.clft_cwt_attn import CWT
from clft_ctca_fusion.clft_ctsa import CTSA
from clft_ctca_fusion.clft_ctca import CTCA
from clft_ctca_fusion.clft_mask import patch_mask_from_lidar, apply_token_mask_to_vit_emb


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
        self.patch_size = patch_size

        # cls token 관련 클래스 객체 생성
        self.cwt = CWT()
        self.ctsa = CTSA(attn_drop=0.0, proj_drop=0.0)
        self.ctca = CTCA(attn_drop=0.0, proj_drop=0.0)

        # g = sigmoid(1x1conv([L, Δ, M]))  -> 토큰별 Linear
        # 입력 채널: [L (D), Δ (D), M (1)] => 2D+1
        # 2D + 1 -> 3D + 1 (gate input에 cam추가)
        # self.gate_proj = nn.Linear(3 * emb_dim + 1, 1, bias=True)

        self.gate_proj = nn.Linear(2 * emb_dim + 1, 1, bias=True)
        
        # cam entropy용
        # L, CTCA, M, U_C
        # self.gate_proj = nn.Linear(2 * emb_dim + 2, 1, bias=True)

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

        # ctca gate logging
        self.last_tokgate = {
            "S2": {"g_mean": None, "g_std": None, "d_norm": None, "upd_norm": None},
            "S0": {"g_mean": None, "g_std": None, "d_norm": None, "upd_norm": None},
        }
        self.last_tokgate_map = {"S2": None, "S0": None}  # (B,1,Hp,Wp)
        # self.alpha = 0.7

        self.alpha = 0.1

        # cam entropy용
        # self.nclasses = nclasses
        # self.entropy_norm = True
        # hidden = emb_dim // 2
        # self.cam_aux_logits = nn.Sequential(
        #     nn.Linear(emb_dim, hidden, bias=True),
        #     nn.GELU,
        #     nn.Linear(hidden, nclasses, bias=True)
        # )

    def _camera_uncertainty(self, logits):
        probs = torch.softmax(logits, dim=-1)

        ent = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        if self.entropy_norm and self.nclasses > 1:
            ent = ent / float(np.log(self.nclasses))

        return ent


    def forward(self, rgb, lidar, modal='rgb', return_extras=False):
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
            mask_tok = patch_mask_from_lidar(lidar, patch_size=self.patch_size, eps=0.0, dtype=torch.float32)  # (B,N,1)

        # mask token을 2d로 바꾸기 위해 grid 사이즈 구하기
        mask_2d = None
        Hp = Wp = None
        if mask_tok is not None:
            B, C, H, W = lidar.shape
            P = self.patch_size
            Hp, Wp = H // P, W // P
            # (B,N,1) -> (B,1,Hp,Wp)
            mask_2d = mask_tok.view(B, Hp, Wp, 1).permute(0, 3, 1, 2).contiguous()  # float(0/1)

        extras = {}
        if modal == 'cross_fusion':
            extras['HpWp'] = (Hp, Wp)

        previous_stage = None

        for i in np.arange(len(self.fusions) - 1, -1, -1):
            hook_to_take = 't' + str(self.hooks[i])

            if modal == 'cross_fusion':
                # 각 모달리티 patch embedding
                activation_result_rgb = self.activation_rgb[hook_to_take]
                activation_result_xyz = self.activation_xyz[hook_to_take]

                reassemble_result_RGB = self.reassembles_RGB[i](activation_result_rgb)
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result_xyz)

                if i == 2:
                    cwt_cam2, w_cam2 = self.cwt(activation_result_rgb)
                    # cwt_xyz2, w_xyz2 = self.cwt(activation_result_xyz)        # mask 미적용

                    # token slot 정규화용
                    extras['tok_cam_s2'] = cwt_cam2
                    extras['w_cam_s2']   = w_cam2
                    # extras['tok_xyz_s2'] = cwt_xyz2
                    # extras['w_xyz_s2']   = w_xyz2

                    # cls_token_stage2 = torch.cat([cwt_cam2, cwt_xyz2], dim=1)   # (B, 2K, D) 같은 형태
                    # ctsa_out2 = self.ctsa(cls_token_stage2)                     # (B, Nk, D)  Nk=2K

                    # ----- CTCA + gated update (LiDAR token update) -----
                    cls_xyz2 = activation_result_xyz[:, :1, :]                  # (B,1,D)  CLS 고정
                    L2 = activation_result_xyz[:, 1:, :]                        # (B,N,D)
                    C2 = activation_result_rgb[:, 1:, :]

                    # delta2 = self.ctca(L2, ctsa_out2)                           # (B,N,D)  Δ, mask 미적용
                    delta2 = self.ctca(L2, cwt_cam2)

                    # 클래스 토큰 concat (ctsa)제거 실험 (라이다 토큰으로만 클래스 토큰 생성)
                    # delta2 = self.ctca(L2, cwt_xyz2)

                    # CTCA heatmap
                    # with torch.no_grad():
                    #     # delta: (B, N, D)
                    #     delta_norm2 = torch.norm(delta2, dim=-1)  # (B, N)
                    #     self.last_delta_norm2 = delta_norm2       # token space
                    #     self.last_delta_hw2   = (Hp, Wp)

                    # g = sigmoid(Linear([L, Δ, M]))
                    if mask_tok is None:
                        raise RuntimeError("mask_tok is None in cross_fusion; check mask creation.")
                    
                    
                    # gate_in2 = torch.cat([L2, delta2, mask_tok], dim=-1)
                    # g2 = torch.sigmoid(self.gate_proj(gate_in2))                # (B,N,1)

                    # logging
                    # with torch.no_grad():
                    #     g_mean = float(g2.mean().item())
                    #     g_std  = float(g2.std(unbiased=False).item())

                    #     # Δ norm
                    #     d_norm = float(delta2.norm(dim=-1).mean().item())

                    #     # 실제 적용된 update norm = ||(g*M)*Δ||
                    #     upd_norm = float(((g2 * mask_tok) * delta2).norm(dim=-1).mean().item())

                    #     self.last_tokgate["S2"]["g_mean"] = g_mean
                    #     self.last_tokgate["S2"]["g_std"]  = g_std
                    #     self.last_tokgate["S2"]["d_norm"] = d_norm
                    #     self.last_tokgate["S2"]["upd_norm"] = upd_norm

                    #     self.last_tokgate_map["S2"] = g2.view(B, Hp, Wp, 1).permute(0,3,1,2).contiguous().detach()


                    
                    # L2p = L2 + (g2 * mask_tok) * delta2                         # (B,N,D)
                    L2p = L2 + self.alpha*(mask_tok * delta2)
                    # L2p = L2 + (mask_tok * delta2)

                    # L2p = L2
                    activation_result_xyz_updated = torch.cat([cls_xyz2, L2p], dim=1)  # (B,1+N,D)

                    # assemble할 때 업데이트된 xyz 토큰 사용
                    reassemble_result_RGB = self.reassembles_RGB[i](activation_result_rgb)
                    reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result_xyz_updated)

                # elif i == 0:
                #     cwt_cam0, w_cam0 = self.cwt(activation_result_rgb)
                #     cwt_xyz0, w_xyz0 = self.cwt(activation_result_xyz)                  # mask 미적용

                #     # token slot 정규화용
                #     extras['tok_cam_s0'] = cwt_cam0
                #     extras['w_cam_s0']   = w_cam0
                #     extras['tok_xyz_s0'] = cwt_xyz0
                #     extras['w_xyz_s0']   = w_xyz0

                #     cls_token_stage0 = torch.cat([cwt_cam0, cwt_xyz0], dim=1)
                #     ctsa_out0 = self.ctsa(cls_token_stage0)

                #     cls_xyz0 = activation_result_xyz[:, :1, :]
                #     L0 = activation_result_xyz[:, 1:, :]
                #     C0 = activation_result_rgb[:, 1:, :]

                #     delta0 = self.ctca(L0, ctsa_out0)                           # mask 미적용

                #     # 클래스 토큰 concat (ctsa)제거 실험 (라이다 토큰으로만 클래스 토큰 생성)
                #     # delta0 = self.ctca(L0, cwt_xyz0)
                    
                #     # CTCA heatmap
                #     # with torch.no_grad():
                #     #     # delta: (B, N, D)
                #     #     delta_norm0 = torch.norm(delta0, dim=-1)  # (B, N)
                #     #     self.last_delta_norm0 = delta_norm0       # token space
                #     #     self.last_delta_hw0   = (Hp, Wp)
                    
                    
                #     gate_in0 = torch.cat([L0, delta0, mask_tok], dim=-1)
                #     g0 = torch.sigmoid(self.gate_proj(gate_in0))

                #     # logging
                #     with torch.no_grad():
                #         g_mean = float(g0.mean().item())
                #         g_std  = float(g0.std(unbiased=False).item())

                #         d_norm = float(delta0.norm(dim=-1).mean().item())
                #         upd_norm = float(((g0 * mask_tok) * delta0).norm(dim=-1).mean().item())

                #         self.last_tokgate["S0"]["g_mean"] = g_mean
                #         self.last_tokgate["S0"]["g_std"]  = g_std
                #         self.last_tokgate["S0"]["d_norm"] = d_norm
                #         self.last_tokgate["S0"]["upd_norm"] = upd_norm

                #         self.last_tokgate_map["S0"] = g0.view(B, Hp, Wp, 1).permute(0,3,1,2).contiguous().detach()


                    
                #     L0p = L0 + (g0 * mask_tok) * delta0
                #     # L0p = L0 + (mask_tok * delta0)
                #     # L0p = L0 + self.alpha*(mask_tok * delta0)

                #     # L0p = L0
                #     activation_result_xyz_updated = torch.cat([cls_xyz0, L0p], dim=1)

                #     reassemble_result_RGB = self.reassembles_RGB[i](activation_result_rgb)
                #     reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result_xyz_updated)

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

            # reassemble_result_XYZ의 해상도와 맞춰주기
            M_stage = None
            if modal == "cross_fusion" and mask_2d is not None:
                Hs, Ws = reassemble_result_XYZ.shape[-2], reassemble_result_XYZ.shape[-1]
                # nearest: 0/1 마스크 보존
                M_stage = torch.nn.functional.interpolate(mask_2d, size=(Hs, Ws), mode="nearest")

            fusion_result = self.fusions[i](reassemble_result_RGB, reassemble_result_XYZ, previous_stage, modal, M_stage)

            # fusion_result = self.fusions[i](reassemble_result_RGB, reassemble_result_XYZ, previous_stage, modal)
            previous_stage = fusion_result

        out_depth = None
        out_segmentation = None
        if self.head_depth is not None:
            out_depth = self.head_depth(previous_stage)
        if self.head_segmentation is not None:
            out_segmentation = self.head_segmentation(previous_stage)


        if return_extras:
            return out_depth, out_segmentation, extras # extras: token slot 정규화용
        else:
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