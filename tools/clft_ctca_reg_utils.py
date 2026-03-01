'''
trainer에서 호출할 utils
'''

# clft_cwt_reg_utils.py
import torch
import torch.nn.functional as F


# -----------------------------
# GT -> patch-level GT utilities
# -----------------------------
@torch.no_grad()
def gt_to_patch_labels(
    gt_hw: torch.Tensor,
    Hp: int,
    Wp: int,
    ignore_id: int = 3,
    mode: str = "nearest",
) -> torch.Tensor:
    """
    Convert pixel GT (B,H,W) to patch GT (B,Hp*Wp).

    Args:
        gt_hw: (B,H,W) integer labels
        Hp, Wp: patch grid size (e.g., H//patch_size, W//patch_size)
        ignore_id: ignore label id (kept as is)
        mode: interpolation mode. For labels use "nearest".

    Returns:
        gt_patch: (B, Hp*Wp) integer labels
    """
    if gt_hw.dim() != 3:
        raise ValueError(f"gt_hw must be (B,H,W), got {tuple(gt_hw.shape)}")

    gt = gt_hw.unsqueeze(1).float()               # (B,1,H,W)
    gt_ds = F.interpolate(gt, size=(Hp, Wp), mode=mode)  # (B,1,Hp,Wp)
    gt_ds = gt_ds.squeeze(1).long()              # (B,Hp,Wp)
    gt_patch = gt_ds.flatten(1)                  # (B,Hp*Wp)
    return gt_patch


def ensure_patch_count(w_slot: torch.Tensor, Hp: int, Wp: int):
    """
    w_slot: (B,K,Np) where Np should equal Hp*Wp.
    """
    B, K, Np = w_slot.shape
    if Np != Hp * Wp:
        raise ValueError(f"w_slot Np={Np} != Hp*Wp={Hp*Wp}. Check patch_size/Hp/Wp alignment.")


# -----------------------------
# Purity loss (slot semantic formation)
# -----------------------------
def cwt_purity_loss(
    w_slot: torch.Tensor,
    gt_patch: torch.Tensor,
    class_ids: list,
    ignore_id: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Purity loss for class-wise tokens.

    Intuition:
      - slot k should place most of its attention mass on patches that belong to class_ids[k].

    Args:
        w_slot:   (B,K,Np) slot-wise patch weights (sum over Np ~= 1 per slot)
        gt_patch: (B,Np) patch labels
        class_ids: list of length K, mapping slot index -> label id
                  e.g., K=2, class_ids=[person_id, car_id]
        ignore_id: ignore label id (excluded)
        eps: numerical stability

    Returns:
        scalar loss

    Skip samples where the target class is absent in gt_patch to avoid
    -log(eps) exploding when class pixels/patches do not exist in the batch sample.
    """
    if w_slot.dim() != 3:
        raise ValueError(f"w_slot must be (B,K,Np), got {tuple(w_slot.shape)}")
    if gt_patch.dim() != 2:
        raise ValueError(f"gt_patch must be (B,Np), got {tuple(gt_patch.shape)}")

    B, K, Np = w_slot.shape
    if len(class_ids) != K:
        raise ValueError(f"class_ids length {len(class_ids)} must equal K={K}")

    if gt_patch.shape[0] != B or gt_patch.shape[1] != Np:
        raise ValueError(f"gt_patch shape {tuple(gt_patch.shape)} must match (B,Np)=({B},{Np})")

    gt = gt_patch
    valid = (gt != ignore_id)

    loss = w_slot.new_tensor(0.0)
    n_terms = 0

    for k in range(K):
        cls_id = int(class_ids[k])
        m = ((gt == cls_id) & valid).float()          # (B,Np)
        present = (m.sum(dim=1) > 0)                  # (B,) bool: class exists in this sample?

        if present.any():
            mass = (w_slot[:, k, :] * m).sum(dim=1)   # (B,)
            loss_vec = -torch.log(mass.clamp_min(eps))
            loss = loss + loss_vec[present].mean()
            n_terms += 1
        # else: skip this class term entirely for this batch (no samples contain the class)

    if n_terms == 0:
        # No class present in any sample for any slot: return 0 to avoid poisoning training
        return w_slot.new_tensor(0.0)

    return loss / n_terms


# -----------------------------
# InfoNCE (token alignment)
# -----------------------------
def infonce_token_alignment_loss(
    tok_a: torch.Tensor,
    tok_b: torch.Tensor,
    temperature: float = 0.07,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    InfoNCE alignment across batch, class-wise.

    Args:
        tok_a: (B,K,D)
        tok_b: (B,K,D)  (same slot order as tok_a)
        temperature: tau
        symmetric: if True, do both a->b and b->a and average

    Returns:
        scalar loss
    """
    if tok_a.dim() != 3 or tok_b.dim() != 3:
        raise ValueError(f"tok_a/tok_b must be (B,K,D). got {tuple(tok_a.shape)} and {tuple(tok_b.shape)}")
    if tok_a.shape != tok_b.shape:
        raise ValueError(f"tok_a shape {tuple(tok_a.shape)} != tok_b shape {tuple(tok_b.shape)}")

    B, K, D = tok_a.shape
    labels = torch.arange(B, device=tok_a.device)

    def _dir_loss(a, b):
        # a,b: (B,D)
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = (a @ b.t()) / max(temperature, 1e-6)  # (B,B)
        return F.cross_entropy(logits, labels)

    loss = 0.0
    for k in range(K):
        loss_k = _dir_loss(tok_a[:, k, :], tok_b[:, k, :])
        if symmetric:
            loss_k = 0.5 * (loss_k + _dir_loss(tok_b[:, k, :], tok_a[:, k, :]))
        loss = loss + loss_k

    return loss / K


# -----------------------------
# Convenience wrapper
# -----------------------------
def cwt_reg_losses(
    tok_cam: torch.Tensor,
    w_cam: torch.Tensor,
    tok_lidar: torch.Tensor,
    w_lidar: torch.Tensor,
    gt_hw: torch.Tensor,
    Hp: int,
    Wp: int,
    class_ids: list,
    ignore_id: int = 3,
    lambda_purity: float = 1.0,
    lambda_infonce: float = 1.0,
    temperature: float = 0.07,
) -> dict:
    """
    One-stop to compute purity + InfoNCE losses.

    Args:
        tok_cam, tok_lidar: (B,K,D)
        w_cam, w_lidar: (B,K,Np)
        gt_hw: (B,H,W)
        Hp,Wp: patch grid size (must match Np)
        class_ids: e.g., [person_id, car_id] (length K)
    """
    ensure_patch_count(w_cam, Hp, Wp)
    ensure_patch_count(w_lidar, Hp, Wp)

    gt_patch = gt_to_patch_labels(gt_hw, Hp, Wp, ignore_id=ignore_id)  # (B,Np)

    purity_cam = cwt_purity_loss(w_cam, gt_patch, class_ids=class_ids, ignore_id=ignore_id)
    purity_lid = cwt_purity_loss(w_lidar, gt_patch, class_ids=class_ids, ignore_id=ignore_id)

    infonce = infonce_token_alignment_loss(tok_cam, tok_lidar, temperature=temperature, symmetric=True)

    out = {
        "loss_purity_cam": purity_cam,
        "loss_purity_lidar": purity_lid,
        "loss_infonce": infonce,
        "loss_reg_total": lambda_purity * (purity_cam + purity_lid) + lambda_infonce * infonce,
    }
    return out