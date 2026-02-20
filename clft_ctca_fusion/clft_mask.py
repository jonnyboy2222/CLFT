# clft_lidar_mask.py
import torch
import torch.nn.functional as F


def pixel_occupancy_from_lidar(lidar_img: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """
    Make a pixel-level occupancy map from LiDAR projection image.

    Args:
        lidar_img: (B, C, H, W) float tensor. Empty pixels are expected to be 0.
                  (This matches your lidar_process.py behavior: X,Y,Z are zero-initialized
                   and only point-projected pixels are filled.)  # see citation below
        eps: threshold; treat |value| > eps as present

    Returns:
        occ: (B, 1, H, W) bool tensor. True where any channel has non-zero (or >eps) value.
    """
    if lidar_img.dim() != 4:
        raise ValueError(f"Expected lidar_img (B,C,H,W), got {tuple(lidar_img.shape)}")

    # present if any channel is non-zero at that pixel
    occ = (lidar_img.abs() > eps).any(dim=1, keepdim=True)  # (B,1,H,W) bool
    return occ


def patch_mask_from_pixel_occupancy(
    occ: torch.Tensor,
    patch_size: int = 16,
    flatten: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert pixel occupancy (B,1,H,W) to patch-token mask aligned with ViT patch tokens.

    Rule: a patch is valid if ANY pixel in its PxP region is occupied.

    Args:
        occ: (B,1,H,W) bool/0-1 tensor
        patch_size: ViT patch size (paper setting: 16)
        flatten: if True, return (B, N, 1). else return (B, Hp, Wp) or (B,1,Hp,Wp)
        dtype: output dtype for mask (float32 recommended for gating)

    Returns:
        m: if flatten=True -> (B, N, 1) where N=(H/P)*(W/P)
           else -> (B, 1, Hp, Wp) float
    """
    if occ.dim() != 4 or occ.size(1) != 1:
        raise ValueError(f"Expected occ (B,1,H,W), got {tuple(occ.shape)}")

    B, _, H, W = occ.shape
    P = patch_size
    if H % P != 0 or W % P != 0:
        raise ValueError(f"H,W must be divisible by patch_size={P}. Got H={H}, W={W}")

    # max-pool over PxP blocks gives OR behavior
    occ_f = occ.to(dtype=torch.float32)
    patch_occ = F.max_pool2d(occ_f, kernel_size=P, stride=P)  # (B,1,Hp,Wp), values in [0,1]

    if not flatten:
        return (patch_occ > 0).to(dtype=dtype)

    # Flatten in row-major order: (Hp,Wp) -> N
    Hp, Wp = H // P, W // P
    m = (patch_occ.view(B, 1, Hp * Wp).transpose(1, 2) > 0).to(dtype=dtype)  # (B,N,1)
    return m


def patch_mask_from_lidar(
    lidar_img: torch.Tensor,
    patch_size: int = 16,
    eps: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convenience wrapper: (B,C,H,W) lidar_img -> (B,N,1) patch-token mask.
    """
    occ = pixel_occupancy_from_lidar(lidar_img, eps=eps)
    return patch_mask_from_pixel_occupancy(occ, patch_size=patch_size, flatten=True, dtype=dtype)


def apply_patch_mask_to_update(delta: torch.Tensor, gate: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Utility for your update rule: (g ⊙ m) ⊙ Δ

    Args:
        delta: (B, N, D)
        gate:  (B, N, 1) or (B, N, D)
        mask:  (B, N, 1)  (output of patch_mask_from_lidar)

    Returns:
        masked_delta: (B, N, D)
    """
    if delta.dim() != 3:
        raise ValueError(f"delta must be (B,N,D), got {tuple(delta.shape)}")
    if mask.dim() != 3:
        raise ValueError(f"mask must be (B,N,1), got {tuple(mask.shape)}")

    # Broadcast to (B,N,D) safely
    masked = delta * gate * mask
    return masked

def apply_token_mask_to_vit_emb(emb: torch.Tensor, mask_tok: torch.Tensor):
    """
    emb: (B, 1+N, D)  (CLS + patch tokens)
    mask_tok: (B, N, 1)  patch token mask (CLS 제외)
    return: (B, 1+N, D)  CLS 고정, patch만 마스킹
    """
    if mask_tok is None:
        return emb

    cls = emb[:, :1, :]     # (B,1,D)
    tok = emb[:, 1:, :]     # (B,N,D)

    # shape check
    if mask_tok.dim() == 2:
        m = mask_tok.unsqueeze(-1)  # (B,N,1)
    else:
        m = mask_tok
    if m.shape[1] != tok.shape[1]:
        raise RuntimeError(f"mask_tok N={m.shape[1]} != token N={tok.shape[1]}")

    m = m.to(device=tok.device, dtype=tok.dtype)  # fp16/amp 안전
    tok = tok * m                                  # hard invalidate

    return torch.cat([cls, tok], dim=1)

