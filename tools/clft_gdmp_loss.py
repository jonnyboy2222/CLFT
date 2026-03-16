import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_like(x: torch.Tensor, size_hw, mode: str, align_corners):
    """입력 텐서를 (H,W)=size_hw로 리사이즈한다."""
    if x.shape[-2:] == tuple(size_hw):
        return x
    return F.interpolate(x, size=size_hw, mode=mode, align_corners=align_corners)


def _resize_gt_maps(
    M_gt: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    size_hw,
):
    """
    GT / weights / valid를 stage 해상도에 맞춰 리사이즈한다.

    철학:
    - M_gt는 0~1 연속값 prior이므로 bilinear(부드럽게)로 downsample.
    - weights/valid는 마스크 성격이 강하므로 nearest로 downsample (경계 번짐 방지).

    Args:
        M_gt:    (B,C,H,W)
        weights: (B,C,H,W)
        valid:   (B,1,H,W)
        size_hw: (Hs, Ws)

    Returns:
        M_s: (B,C,Hs,Ws)
        W_s: (B,C,Hs,Ws)
        V_s: (B,1,Hs,Ws)
    """
    M_s = _resize_like(M_gt, size_hw, mode="bilinear", align_corners=False)
    W_s = _resize_like(weights, size_hw, mode="nearest", align_corners=None)
    V_s = _resize_like(valid, size_hw, mode="nearest", align_corners=None)
    return M_s, W_s, V_s


def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    valid: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    가중 MSE (분모를 weight 합으로 정규화).

    Args:
        pred/target/weight: (B,C,H,W)
        valid(optional):    (B,1,H,W) 또는 (B,C,H,W)

    Notes:
        - weight는 채널별 충돌 방지(다른 클래스는 0) + fg/bg 비중 조절에 사용
        - valid는 ignore 등 완전 제외용 (0이면 loss=0)
        - 픽셀 수/객체 수가 이미지마다 달라도 scale이 크게 흔들리지 않도록
          den=weight.sum()으로 정규화한다.
    """
    if valid is not None:
        # valid가 (B,1,H,W)면 채널 축으로 broadcast
        if valid.shape[1] == 1 and pred.shape[1] != 1:
            valid = valid.expand(-1, pred.shape[1], -1, -1)
        weight = weight * valid

    diff2 = (pred - target).pow(2)
    num = (weight * diff2).sum()
    den = weight.sum().clamp_min(eps)
    return num / den


class GDMPLoss(nn.Module):
    """
    PGDP 멀티스케일 loss 계산기.

    전제:
      - PGDP는 p2,p1,p0를 (B,C,Hs,Ws)로 반환 (C=클래스 채널 수)
      - GT 빌더는 anno 해상도에서 M_gt, weights, valid를 반환
        M_gt: (B,C,H,W), weights: (B,C,H,W), valid: (B,1,H,W)

    사용법:
      total, logs = loss_fn(p2,p1,p0, gt_dict)
    """

    def __init__(self, w2: float = 1.0, w1: float = 1.0, w0: float = 1.0):
        super().__init__()
        self.w2 = float(w2)
        self.w1 = float(w1)
        self.w0 = float(w0)

    def forward(self, p2: torch.Tensor, p1: torch.Tensor, p0: torch.Tensor, gt: dict):
        """
        Args:
            p2,p1,p0: (B,C,Hs,Ws) sigmoid 출력(0~1)
            gt: {
                "M_gt": (B,C,H,W),
                "weights": (B,C,H,W),
                "valid": (B,1,H,W),
            }

        Returns:
            total_loss: scalar tensor
            logs: dict (detach된 텐서들)
        """
        M_gt = gt["M_gt"]
        weights = gt["weights"]
        valid = gt["valid"]

        # weights는 (B,C,H,W). fg는 bg_weight보다 훨씬 큰 값(기본 fg_weight=10).
        # valid까지 고려해서 fg 판단
        w_eff = weights * valid  # (B,C,H,W) broadcast
        # 샘플별 fg 픽셀 존재 여부: (B,)
        has_fg = (w_eff > 1.0).flatten(1).any(dim=1)  # threshold는 fg_weight(10) vs bg_weight(0.1) 고려
        if not has_fg.any():
            # 배치 전체가 fg 없음이면 loss=0
            zero = p0.new_tensor(0.0)
            logs = {"pgdp/l2": zero, "pgdp/l1": zero, "pgdp/l0": zero, "pgdp/total": zero}
            return zero, logs

        # keep 되는 샘플만 슬라이스
        M_gt = M_gt[has_fg]
        weights = weights[has_fg]
        valid = valid[has_fg]
        p2 = p2[has_fg]
        p1 = p1[has_fg]
        p0 = p0[has_fg]

        # stage별 GT 리사이즈(고해상도 1회 생성 → 각 stage로 downsample)
        M2, W2, V2 = _resize_gt_maps(M_gt, weights, valid, p2.shape[-2:])
        M1, W1, V1 = _resize_gt_maps(M_gt, weights, valid, p1.shape[-2:])
        M0, W0, V0 = _resize_gt_maps(M_gt, weights, valid, p0.shape[-2:])

        # 가중 MSE
        l2 = weighted_mse(p2, M2, W2, V2)
        l1 = weighted_mse(p1, M1, W1, V1)
        l0 = weighted_mse(p0, M0, W0, V0)

        total = self.w2 * l2 + self.w1 * l1 + self.w0 * l0

        logs = {
            "pgdp/l2": l2.detach(),
            "pgdp/l1": l1.detach(),
            "pgdp/l0": l0.detach(),
            "pgdp/total": total.detach(),
        }
        return total, logs
    


def dice_loss_softmax(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_id: int,
    ignore_index: int = None,
    eps: float = 1e-6,
):
    """
    Soft Dice loss for a single class (semantic segmentation).

    logits: (B, C, H, W)
    target: (B, H, W)
    class_id: dice를 계산할 클래스 id (예: human 클래스 id)
    """

    B, C, H, W = logits.shape

    # ignore mask
    if ignore_index is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        valid = target != ignore_index


    # 해당 클래스 GT 마스크
    g = (target == class_id) & valid  # (B, H, W)

    # 이 클래스가 존재하는 샘플만 계산
    has = g.view(B, -1).any(dim=1)
    if not has.any():
        return logits.new_tensor(0.0)

    prob = torch.softmax(logits, dim=1)[:, class_id]  # (B,H,W)

    prob = prob[has]
    g = g[has].float()

    intersection = (prob * g).sum(dim=(1, 2))
    union = prob.sum(dim=(1, 2)) + g.sum(dim=(1, 2))

    dice = (2 * intersection + eps) / (union + eps)

    return 1.0 - dice.mean()