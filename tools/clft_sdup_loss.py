import torch
import torch.nn.functional as F

def _normalize_map01(x, eps=1e-6):
    # x: (B,1,H,W)
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def _build_sdup_target(output_seg, p0_rgb, u_map_rgb):
    """
    output_seg : (B,C,H,W) logits
    p0_rgb     : (B,2,H0,W0) PGDP output
    u_map_rgb  : (B,1,H0,W0) SDUP prediction (only for target size reference)
    return     : target_u (B,1,H0,W0), detached
    """
    eps = 1e-6

    # 1) segmentation entropy
    prob = torch.softmax(output_seg, dim=1)  # (B,C,H,W)
    entropy = -(prob * torch.log(prob.clamp_min(eps))).sum(dim=1, keepdim=True)  # (B,1,H,W)

    # normalize by log(C)
    num_classes = output_seg.shape[1]
    entropy = entropy / torch.log(torch.tensor(float(num_classes), device=output_seg.device))

    # 2) resize to SDUP map resolution
    entropy_s0 = F.interpolate(
        entropy,
        size=u_map_rgb.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    # 3) PGDP foreground prior
    # p0_rgb: (B,2,H0,W0) -> object prior (max over 2 channels)
    p0_fg = p0_rgb.max(dim=1, keepdim=True).values

    # 4) normalize both, then combine
    entropy_s0 = _normalize_map01(entropy_s0)
    p0_fg = _normalize_map01(p0_fg)

    target_u = entropy_s0 * p0_fg

    # optional: sharpen a bit if needed later
    # target_u = target_u.pow(0.75)

    return target_u.detach()