import numpy as np
import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from clft.reassemble import Reassemble
from clft.fusion import Fusion
from clft.head import HeadDepth, HeadSeg, HeadAuxHuman

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

        # Register hooks
        self.activation = {}
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

        # 보조헤드 v3, v4 공통
        # self.aux_lateral = nn.Sequential(
        #     # stage1에서 stage0로 upsample
        #     nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     # allign_corners=False로 변경
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        # )

        # ups v1
        # stage1용
        # self.aux_lateral = nn.Sequential(
        #     nn.Conv2d(resample_dim, resample_dim, 3, padding=1, groups=resample_dim, bias=True),
        #     nn.Conv2d(resample_dim, resample_dim, 1, bias=True),
        #     nn.ReLU(inplace=True),

        #     # allign_corners=False로 변경
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        # )

        # stage2용 upsample 경로 추가
        # self.aux_lateral2 = nn.Sequential(
        #     nn.Conv2d(resample_dim, resample_dim, 3, padding=1, groups=resample_dim, bias=True),
        #     nn.Conv2d(resample_dim, resample_dim, 1, bias=True),
        #     nn.ReLU(inplace=True),

        #     # allign_corners=False로 변경
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),

        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        # )

        # ups v2
        # 모든 conv -> dw + pw
        # stage1용
        # self.aux_lateral = nn.Sequential(
        #     nn.Conv2d(resample_dim, resample_dim, 3, padding=1, groups=resample_dim, bias=True),
        #     nn.Conv2d(resample_dim, resample_dim, 1, bias=True),
        #     nn.ReLU(inplace=True),

        #     # allign_corners=False로 변경
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(resample_dim, resample_dim, 3, padding=1, groups=resample_dim, bias=True),
        #     nn.Conv2d(resample_dim, resample_dim, 1, bias=True),
        #     nn.ReLU(inplace=True),
        # )

        # # stage2용 upsample 경로 추가
        # self.aux_lateral2 = nn.Sequential(
        #     nn.Conv2d(resample_dim, resample_dim, 3, padding=1, groups=resample_dim, bias=True),
        #     nn.Conv2d(resample_dim, resample_dim, 1, bias=True),
        #     nn.ReLU(inplace=True),

        #     # allign_corners=False로 변경
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(resample_dim, resample_dim, 3, padding=1, groups=resample_dim, bias=True),
        #     nn.Conv2d(resample_dim, resample_dim, 1, bias=True),
        #     nn.ReLU(inplace=True),

        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(resample_dim, resample_dim, 3, padding=1, groups=resample_dim, bias=True),
        #     nn.Conv2d(resample_dim, resample_dim, 1, bias=True),
        #     nn.ReLU(inplace=True),
        # )

        # ups v2
        # main head용 upsample 후 정렬
        # self.main_lateral = nn.Sequential(
        #     nn.Conv2d(resample_dim, resample_dim, 3, padding=1, groups=resample_dim, bias=True),
        #     nn.Conv2d(resample_dim, resample_dim, 1, bias=True),
        #     nn.ReLU(inplace=True)
        # )

        # ups v3
        # stage1용
        # self.aux_lateral = nn.Sequential(
        #     nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),

        #     # allign_corners=False로 변경
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        # )

        # # stage2용 upsample 경로 추가
        # self.aux_lateral2 = nn.Sequential(
        #     nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),

        #     # allign_corners=False로 변경
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),

        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        # )

        # ups v3
        # main head용 upsample 후 정렬
        # self.main_lateral = nn.Sequential(
        #     nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True)
        # )

        # 보조헤드 v5
        # self.aux_proj = nn.Conv2d(resample_dim, resample_dim, kernel_size=1)

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
            # 보조헤드
            self.head_aux_human = HeadAuxHuman(resample_dim)

    def forward(self, rgb, lidar, modal='rgb'):
        t = self.transformer_encoders(lidar)
        previous_stage = None
        
        # 보조헤드 v3
        # stage1 = None
        # stage0 = None

        # 보조헤드 v4
        # stage2 = None
        # stage1 = None

        # 보조헤드 v5
        # stage2 = None
        # stage0 = None

        for i in np.arange(len(self.fusions)-1, -1, -1):
            hook_to_take = 't'+str(self.hooks[i])
            activation_result = self.activation[hook_to_take]
            if modal == 'rgb':
                reassemble_result_RGB = self.reassembles_RGB[i](activation_result) #claude check here
                reassemble_result_XYZ = torch.zeros_like(reassemble_result_RGB) # this is just to keep the space allocated but it will not be used later in fusion
            if modal == 'lidar':
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result) #claude check here
                reassemble_result_RGB = torch.zeros_like(reassemble_result_XYZ) # this is just to keep the space allocated but it will not be used later in fusion
            if modal == 'cross_fusion':
                reassemble_result_RGB = self.reassembles_RGB[i](activation_result) #claude check here
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result) #claude check here
            
            fusion_result = self.fusions[i](reassemble_result_RGB, reassemble_result_XYZ, previous_stage, modal) #claude check here
            previous_stage = fusion_result

            # 보조헤드 v3
            # if i==1:
            #     stage1 = fusion_result
            # if i==0:
            #     stage0 = fusion_result

            # 보조헤드 v4
            # if i==2:
            #     stage2 = fusion_result
            # if i==1:
            #     stage1 = fusion_result

            # 보조헤드 v5
            # if i==2:
            #     stage2 = fusion_result
            # if i==0:
            #     stage0 = fusion_result

        #dbg
        # print(f"stage2 : {stage2.shape}")
        # print(f"stage1 : {stage1.shape}")
        # print(f"stage0 : {stage0.shape}")

        out_depth = None
        out_segmentation = None
        # 보조헤드
        # out_aux_human = None

        if self.head_depth != None:
            out_depth = self.head_depth(previous_stage)
        if self.head_segmentation != None:
            # 기존
            out_segmentation = self.head_segmentation(previous_stage)

            # ups v2
            # main head에 들어가는 input에 정렬용 conv 적용
            # last_stage = previous_stage
            # main_head_input = self.main_lateral(last_stage)
            # out_segmentation = self.head_segmentation(main_head_input)

        # 보조헤드
        # if hasattr(self, "head_aux_human") and self.head_aux_human is not None:
        #     # 보조헤드 v3
        #     aux_input = previous_stage # fallback
        #     # if (stage1 is not None) and (stage0 is not None):
        #     #     aux_input = self.aux_lateral(stage1) + stage0
        #     # v4에서 stage2 + stage1 사용
        #     if (stage2 is not None) and (stage1 is not None):
        #         # up_stage2 = self.aux_lateral(self.aux_lateral(stage2))
        #         # up_stage1 = self.aux_lateral(stage1)

        #         # ups v1
        #         up_stage2 = self.aux_lateral2(stage2)
        #         up_stage1 = self.aux_lateral(stage1)
        #         aux_input = up_stage2 + up_stage1
            # v5 stage2 + stage0
            # if (stage2 is not None) and (stage0 is not None):
            #     up_stage2 = self.aux_lateral(self.aux_lateral(stage2))
            #     proj_stage0 = self.aux_proj(stage0)
            #     aux_input = up_stage2 + proj_stage0
            # v3에서 previous_stage -> aux_input 변경
            # out_aux_human = self.head_aux_human(aux_input)

        return out_depth, out_segmentation # , out_aux_human

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        # 기존
        for h in hooks:
            # 원래 주석
            # self.transformer_encoders.layers[h].register_forward_hook(get_activation('t'+str(h)))
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t'+str(h)))
        
        # swin용
        # Swin 계열(backbone.layers[*].blocks[*])인 경우
        # if hasattr(self.transformer_encoders, "layers") and hasattr(self.transformer_encoders.layers[0], "blocks"):
        #     # config["CLFT"]["hooks"]를 stage index로 해석: 예) [0,1,2,3]
        #     for h in hooks:
        #         stage = self.transformer_encoders.layers[h]
        #         block = stage.blocks[-1]  # 해당 stage의 마지막 block에서 feature 추출
        #         block.register_forward_hook(get_activation('t' + str(h)))
        # else:
        #     # ViT 계열: 기존 코드 (blocks[h]) 그대로 사용
        #     for h in hooks:
        #         self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t' + str(h)))
