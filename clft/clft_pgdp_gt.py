import torch
import numpy as np
import cv2


def _to_bhw_long(anno: torch.Tensor) -> torch.Tensor:
    """
    입력 anno 허용 shape:
      (H,W), (1,H,W), (B,H,W), (B,1,H,W)
    반환:
      (B,H,W) long
    """
    if anno.dim() == 2:
        anno = anno.unsqueeze(0)  # (1,H,W)
    elif anno.dim() == 3:
        pass  # (1,H,W) or (B,H,W)
    elif anno.dim() == 4:
        if anno.size(1) == 1:
            anno = anno[:, 0]     # (B,H,W)
        else:
            raise ValueError(f"anno with 4 dims must be (B,1,H,W). Got {tuple(anno.shape)}")
    else:
        raise ValueError(f"Unsupported anno shape: {tuple(anno.shape)}")

    return anno.long()


def _boundary_from_binary(mask_u8: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    binary mask(0/1, uint8)에서 boundary(경계) 픽셀만 추출.
    boundary = mask - erode(mask)
    """
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)

    if mask_u8.max() == 0:
        return mask_u8  # 전부 0이면 boundary도 0

    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    er = cv2.erode(mask_u8, kernel, iterations=1)
    bd = (mask_u8 & (1 - er)).astype(np.uint8)

    # 매우 얇은 객체(1px 선 등)는 erosion으로 전부 사라질 수 있음.
    # 이 경우 boundary가 비어버리면 EDT seed가 없어지므로, boundary를 mask 자체로 둔다.
    if bd.max() == 0:
        bd = mask_u8.copy()

    return bd


def build_pgdp_gt_from_edt(
    anno: torch.Tensor,
    class_ids=(1, 2),
    background_id: int = 0,
    ignore_id: int = 4,
    fg_weight: float = 10.0,
    bg_weight: float = 0.1,
    eps: float = 1e-6,
    boundary_ksize: int = 3,
):
    """
    Semantic-only PGDP GT 생성 (instance/bbox 없음).

    철학:
    - 클래스(객체) 내부에서 '자기 경계까지의 유클리드 거리(EDT)'를 구해 center-prior로 사용
    - 각 connected component(객체)별로 max로 정규화하여 크기 편향을 줄임
      (큰 객체만 center가 강해지는 문제 완화)
    - ignore(id==ignore_id)는 거리 계산 seed로도 사용하지 않고, loss weight=0으로 제외

    반환:
      M_gt    : (B,C,H,W) float, boundary=0, center에 가까울수록 1
      weights : (B,C,H,W) float, 해당 클래스 fg + true background만 약하게, 나머지 0
      valid   : (B,1,H,W) float, ignore 제외 마스크
    """
    anno_bhw = _to_bhw_long(anno)
    device = anno_bhw.device
    B, H, W = anno_bhw.shape
    C = len(class_ids)

    # ignore는 전역적으로 유효 픽셀에서 제외
    valid = (anno_bhw != ignore_id).float().unsqueeze(1)  # (B,1,H,W)

    anno_cpu = anno_bhw.detach().to("cpu").numpy().astype(np.int32)

    M_list = []
    W_list = []

    for b in range(B):
        a = anno_cpu[b]  # (H,W)

        m_channels = []
        w_channels = []

        # true background (anno==background_id)만 background로 인정
        is_bg = (a == background_id)
        is_ignore = (a == ignore_id)

        for cid in class_ids:
            # 해당 클래스 픽셀 (ignore 제외)
            cls_mask = (a == cid)
            cls_mask_valid = cls_mask & (~is_ignore)

            m = np.zeros((H, W), dtype=np.float32)

            if cls_mask_valid.any():
                # (1) 클래스 마스크에서 connected component(객체) 분리
                # cv2.connectedComponents는 0을 배경으로 보고 1..K를 객체 라벨로 부여
                cls_u8 = cls_mask_valid.astype(np.uint8)
                num, lab = cv2.connectedComponents(cls_u8, connectivity=8)

                # (2) 클래스 전체에 대해 "자기 경계까지 거리" EDT를 한 번 계산
                # 경계 픽셀을 0(seed)로 두고, 나머지를 1로 두면
                # cv2.distanceTransform이 "각 픽셀 -> nearest boundary" 거리를 준다.
                boundary = _boundary_from_binary(cls_u8, ksize=boundary_ksize)  # 0/1
                dt_input = np.ones((H, W), dtype=np.uint8)
                dt_input[boundary == 1] = 0  # boundary가 seed(0)

                dt = cv2.distanceTransform(dt_input, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)

                # (3) 객체(connected component)별로 dt를 max 정규화하여 center-prior 생성
                for k in range(1, num):
                    comp = (lab == k)
                    if not comp.any():
                        continue
                    d = dt[comp]
                    dmax = float(d.max())

                    # tiny/line 객체는 dmax≈0일 수 있음 → center prior를 1로 처리
                    if dmax < eps:
                        m[comp] = 1.0
                    else:
                        m[comp] = (d / (dmax + eps)).astype(np.float32)

            # (4) 채널별 weight (충돌 방지 철학 유지)
            # - 해당 클래스 fg: 크게
            # - true background: 작게(완전 무시하면 regression이 불안정해질 수 있어 약하게 둠)
            # - 다른 클래스: 0 (채널 간 목표 충돌 방지)
            # - ignore: 0
            w = np.zeros((H, W), dtype=np.float32)
            w[is_bg] = float(bg_weight)
            w[cls_mask] = float(fg_weight)   # ignore 포함 cls_mask라도 아래에서 valid로 제거됨
            w[is_ignore] = 0.0

            # valid로 한 번 더 안전장치
            w = w * (1.0 - is_ignore.astype(np.float32))

            m_channels.append(m)
            w_channels.append(w)

        M_list.append(np.stack(m_channels, axis=0))  # (C,H,W)
        W_list.append(np.stack(w_channels, axis=0))  # (C,H,W)

    M_gt = torch.from_numpy(np.stack(M_list, axis=0)).to(device=device, dtype=torch.float32)      # (B,C,H,W)
    weights = torch.from_numpy(np.stack(W_list, axis=0)).to(device=device, dtype=torch.float32)  # (B,C,H,W)

    # ignore는 항상 제외(혹시 weight 생성 로직이 바뀌더라도 안전)
    weights = weights * valid

    return M_gt, weights, valid


class PGDPBuilderGT:
    """
    Semantic-only PGDP GT 빌더 (EDT 기반).

    - instance/bbox 없이 semantic mask로부터 class-wise center prior 생성
    - 클래스 내부에서 "자기 경계까지의 EDT"를 이용해 부드러운 위치 힌트를 만듦
    - connected component(객체)별 정규화로 크기 편향 완화
    - ignore(id==4)는 GT/LOSS에서 제외
    """
    def __init__(
        self,
        class_ids=(1, 2),
        background_id=0,
        ignore_id=4,
        fg_weight=10.0,
        bg_weight=0.1,
        boundary_ksize=3,
        eps=1e-6,
    ):
        self.class_ids = tuple(class_ids)
        self.background_id = int(background_id)
        self.ignore_id = int(ignore_id)
        self.fg_weight = float(fg_weight)
        self.bg_weight = float(bg_weight)
        self.boundary_ksize = int(boundary_ksize)
        self.eps = float(eps)

    def __call__(self, anno: torch.Tensor):
        M_gt, weights, valid = build_pgdp_gt_from_edt(
            anno=anno,
            class_ids=self.class_ids,
            background_id=self.background_id,
            ignore_id=self.ignore_id,
            fg_weight=self.fg_weight,
            bg_weight=self.bg_weight,
            boundary_ksize=self.boundary_ksize,
            eps=self.eps,
        )
        return {
            "M_gt": M_gt,        # (B,C,H,W)
            "weights": weights,  # (B,C,H,W)
            "valid": valid,      # (B,1,H,W)
        }
