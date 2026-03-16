import torch
import torch.nn.functional as F

def build_patch_gt_soft(
    gt: torch.Tensor,          # (B,H,W), int64
    patch_size: int,
    class_ids=(1, 2),         # 예: person=1, car=2
    ignore_index: int = 255,
):
    """
    return:
      patch_gt   : (B, Np, C) soft proportion target
      class_mask : (B, C) 각 class가 sample에 충분히 존재하는지
    """
    B, H, W = gt.shape
    assert H % patch_size == 0 and W % patch_size == 0

    Hp, Wp = H // patch_size, W // patch_size
    C = len(class_ids)

    # ignore는 계산에서 제외하기 위한 valid mask
    valid = (gt != ignore_index).float()  # (B,H,W)

    patch_targets = []
    class_present = []

    for cls_id in class_ids:
        cls_map = (gt == cls_id).float() * valid  # ignore 제외

        # patch 내부 class pixel count
        cls_patch = cls_map.view(B, Hp, patch_size, Wp, patch_size)
        cls_patch = cls_patch.permute(0, 1, 3, 2, 4).contiguous()  # (B,Hp,Wp,ps,ps)
        cls_count = cls_patch.sum(dim=(-1, -2))  # (B,Hp,Wp)

        # patch 내부 valid pixel count
        valid_patch = valid.view(B, Hp, patch_size, Wp, patch_size)
        valid_patch = valid_patch.permute(0, 1, 3, 2, 4).contiguous()
        valid_count = valid_patch.sum(dim=(-1, -2)).clamp_min(1.0)  # (B,Hp,Wp)

        # 비율
        cls_ratio = cls_count / valid_count  # (B,Hp,Wp)
        patch_targets.append(cls_ratio)

        # sample 전체에서 해당 class 존재 여부
        class_present.append((cls_count.sum(dim=(1, 2)) > 0).float())  # (B,)

    patch_gt = torch.stack(patch_targets, dim=-1)   # (B,Hp,Wp,C)
    patch_gt = patch_gt.view(B, Hp * Wp, C)         # (B,Np,C)

    class_mask = torch.stack(class_present, dim=-1) # (B,C)

    return patch_gt, class_mask