#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PGDP GT verify + refine (NO argparse)
- Uses the SAME Dataset(list-file) style as pnp_viz.py / trainer pipeline.
- For each sample:
    1) Load rgb/lidar/anno
    2) Build ORIGINAL PGDP GT (trainer와 동일: tools.clft_pgdp_gt.PGDPBuilderGT)
    3) Build REFINED PGDP GT (closing + hole fill + optional border ignore)
    4) Save visualizations:
        - rgb
        - anno (color)
        - pgdp_gt_orig_gray
        - pgdp_gt_refined_gray
        - diff (refined - orig)
        - overlays (orig/refined)

You only edit the "USER CONFIG" section. No arguments.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.dataset import Dataset
from tools.clft_gdmp_gt import GDMPBuilderGT as GDMPBuilderGT_ORIG


# ==========================================================
# USER CONFIG (EDIT HERE)
# ==========================================================
CONFIG_PATH = "/home/john/dev_ws/CLFT/config.json"

# SPLIT is only used by Dataset internally (same as your other scripts)
SPLIT = "val"

# Use any of your lists: all.txt / early_stop_valid.txt / test_*.txt / train_all.txt
LIST_PATH = "/home/john/dev_ws/CLFT/waymo_dataset/splits_clft/train_all.txt"   # <-- 바꿔도 됨
# 예: LIST_PATH = "/mnt/data/test_night_rain.txt"

# Which indices to inspect
START_INDEX = 0
NUM_SAMPLES = 20

# Output directory
OUT_DIR = Path("./pgdp_gt_verify_refine_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target semantic class ids (CLFT: 0 bg, 1 car, 2 human, 4 ignore)
CLASS_IDS = (1, 2)
BACKGROUND_ID = 0
IGNORE_ID = 4

# --- Refinement knobs ---
# 1) close to connect broken GT inside an object (lidar holes / discontinuities)
CLOSE_K = 7           # odd
CLOSE_ITERS = 1

# 2) optional mild dilation BEFORE closing (connect tiny gaps)
USE_DILATE_BEFORE_CLOSE = True
DILATE_K = 3          # odd
DILATE_ITERS = 1

# 3) remove tiny fragments (islands) after refine
MIN_COMPONENT_AREA = 30

# 4) optional border suppression (stop supervising borders that tend to be noisy)
APPLY_BORDER_ZERO = True
BORDER_PX = 6         # e.g., 6~12

# PGDP GT parameters (same semantics as original)
FG_WEIGHT = 10.0
BG_WEIGHT = 0.1
BOUNDARY_KSIZE = 3
EPS = 1e-6

# Visualization knobs
OVERLAY_ALPHA = 0.35
COLORMAP = cv2.COLORMAP_TURBO
HEAT_GAIN = 1.2
# ==========================================================


def robust_norm01(x: np.ndarray, p_low=1.0, p_high=99.0, eps=1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)


def to_gray_png(arr01_hw: np.ndarray) -> np.ndarray:
    arr = np.clip(arr01_hw, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def to_heatmap(arr01_hw: np.ndarray) -> np.ndarray:
    arr = np.clip(arr01_hw, 0.0, 1.0)
    heat = cv2.applyColorMap((arr * 255).astype(np.uint8), COLORMAP)
    heat = np.clip(heat.astype(np.float32) * float(HEAT_GAIN), 0, 255).astype(np.uint8)
    return heat


def overlay(base_bgr: np.ndarray, heat_bgr: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))
    return cv2.addWeighted(base_bgr, 1.0 - alpha, heat_bgr, alpha, 0)


def rgb_tensor_to_bgr(rgb: torch.Tensor) -> np.ndarray:
    # rgb: (3,H,W) float [0,1]
    img = rgb.permute(1, 2, 0).detach().cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def colorize_anno(anno_hw: np.ndarray) -> np.ndarray:
    """
    quick visualization:
      0 bg  -> black
      1 car -> blue-ish
      2 hum -> green-ish
      4 ign -> gray
    """
    H, W = anno_hw.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[anno_hw == 1] = (255, 0, 0)     # BGR
    out[anno_hw == 2] = (0, 255, 0)
    out[anno_hw == 4] = (128, 128, 128)
    return out


def boundary_from_binary(mask_u8: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    boundary = mask - erode(mask)
    """
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    if mask_u8.max() == 0:
        return mask_u8
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    er = cv2.erode(mask_u8, kernel, iterations=1)
    bd = (mask_u8 & (1 - er)).astype(np.uint8)
    if bd.max() == 0:
        bd = mask_u8.copy()
    return bd


def fill_holes(mask_u8_01: np.ndarray) -> np.ndarray:
    """
    Robust hole filling for binary mask (0/1).

    Why needed:
      기존 floodFill(seed=(0,0)) 방식은 객체가 테두리/코너에 닿아 (0,0)이 배경이 아니게 되면
      '전체 이미지가 hole'처럼 오판되어 마스크가 full-image로 폭발할 수 있음.
      (네 로그의 ref_mean≈0.94 케이스)

    Fix:
      1픽셀 background padding을 두르고 floodFill을 수행 -> (0,0)이 항상 background가 됨.
      floodFill로 border-connected background만 표시한 뒤,
      enclosed hole만 채움.
    """
    m = (mask_u8_01 > 0).astype(np.uint8)  # 0/1

    # pad with guaranteed background border
    m_pad = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)

    inv = (1 - m_pad).astype(np.uint8) * 255
    h, w = inv.shape

    flood = inv.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)

    # (0,0) is guaranteed background due to padding
    cv2.floodFill(flood, ffmask, (0, 0), 128)

    bg_border = (flood == 128).astype(np.uint8)  # background connected to border
    holes = ((inv == 255).astype(np.uint8) & (1 - bg_border)).astype(np.uint8)

    filled = ((m_pad == 1).astype(np.uint8) | holes).astype(np.uint8)

    # crop padding
    filled = filled[1:-1, 1:-1]
    return filled


def refine_semantic_mask(a_hw: np.ndarray, cid: int) -> np.ndarray:
    """
    Refine a single class mask (0/1 uint8) safely.

    Pipeline:
      - (optional) dilate
      - close
      - fill holes (robust)
      - remove tiny connected components
      - SAFETY GUARD: if area explodes, rollback to original

    Return:
      keep: (H,W) 0/1 uint8
    """
    is_ignore = (a_hw == IGNORE_ID)
    m0 = ((a_hw == cid) & (~is_ignore)).astype(np.uint8)  # 0/1

    if m0.max() == 0:
        return m0

    orig_area = int(m0.sum())
    m = m0.copy()

    # optional dilate -> close (your existing knobs)
    if USE_DILATE_BEFORE_CLOSE:
        k = np.ones((DILATE_K, DILATE_K), dtype=np.uint8)
        m = cv2.dilate(m, k, iterations=int(DILATE_ITERS))

    k = np.ones((CLOSE_K, CLOSE_K), dtype=np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=int(CLOSE_ITERS))

    # robust hole fill
    m = fill_holes(m)

    # remove tiny fragments
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m, dtype=np.uint8)
    for k_id in range(1, num):
        area = int(stats[k_id, cv2.CC_STAT_AREA])
        if area >= int(MIN_COMPONENT_AREA):
            keep[lab == k_id] = 1

    # -------------------------
    # SAFETY GUARD: prevent full-image explosion
    # -------------------------
    ref_area = int(keep.sum())
    H, W = keep.shape

    # conservative thresholds for verification
    max_growth = 3.0     # allow up to 3x area growth vs original
    max_abs_frac = 0.35  # or at most 35% of the whole image

    if ref_area > int(max_growth * max(1, orig_area)) or (ref_area / float(H * W) > max_abs_frac):
        # rollback to original if explosion detected
        return m0

    return keep


def build_pgdp_gt_refined_from_edt(anno_bhw: torch.Tensor):
    """
    Refined GT:
    - refine class masks (close + fill holes)
    - connected components on refined mask
    - EDT to boundary seeds
    - optional border zeroing (to avoid edge noise)
    """
    if anno_bhw.ndim == 4 and anno_bhw.size(1) == 1:
        anno_bhw = anno_bhw[:, 0]
    assert anno_bhw.ndim == 3, f"Expected (B,H,W), got {tuple(anno_bhw.shape)}"

    device = anno_bhw.device
    B, H, W = anno_bhw.shape
    C = len(CLASS_IDS)

    anno_cpu = anno_bhw.detach().to("cpu").numpy().astype(np.int32)

    # valid: ignore 제외
    valid = (anno_bhw != IGNORE_ID).float().unsqueeze(1)  # (B,1,H,W)

    M_list, W_list = [], []

    # border mask (optional)
    if APPLY_BORDER_ZERO and BORDER_PX > 0:
        border = np.zeros((H, W), dtype=np.float32)
        border[:BORDER_PX, :] = 1
        border[-BORDER_PX:, :] = 1
        border[:, :BORDER_PX] = 1
        border[:, -BORDER_PX:] = 1
        border_keep = 1.0 - border  # 1 inside, 0 border
    else:
        border_keep = np.ones((H, W), dtype=np.float32)

    for b in range(B):
        a = anno_cpu[b]
        is_bg = (a == BACKGROUND_ID)
        is_ignore = (a == IGNORE_ID)

        m_channels, w_channels = [], []

        for cid in CLASS_IDS:
            m = np.zeros((H, W), dtype=np.float32)

            # refined mask
            cls_u8 = refine_semantic_mask(a, cid)  # 0/1 uint8

            if cls_u8.max() > 0:
                # CC on refined
                num, lab = cv2.connectedComponents(cls_u8, connectivity=8)

                # EDT to boundary (seed = boundary pixels)
                boundary = boundary_from_binary(cls_u8, ksize=int(BOUNDARY_KSIZE))
                dt_input = np.ones((H, W), dtype=np.uint8)
                dt_input[boundary == 1] = 0
                dt = cv2.distanceTransform(dt_input, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)

                for k_id in range(1, num):
                    comp = (lab == k_id)
                    if not comp.any():
                        continue
                    d = dt[comp]
                    dmax = float(d.max())
                    if dmax < EPS:
                        m[comp] = 1.0
                    else:
                        m[comp] = (d / (dmax + EPS)).astype(np.float32)

            # weights (same philosophy)
            w = np.zeros((H, W), dtype=np.float32)
            w[is_bg] = float(BG_WEIGHT)
            w[a == cid] = float(FG_WEIGHT)
            w[is_ignore] = 0.0

            # apply valid + border keep
            w = w * (1.0 - is_ignore.astype(np.float32))
            w = w * border_keep.astype(np.float32)
            m = m * border_keep.astype(np.float32)

            m_channels.append(m)
            w_channels.append(w)

        M_list.append(np.stack(m_channels, axis=0))
        W_list.append(np.stack(w_channels, axis=0))

    M_gt = torch.from_numpy(np.stack(M_list, axis=0)).to(device=device, dtype=torch.float32)      # (B,C,H,W)
    weights = torch.from_numpy(np.stack(W_list, axis=0)).to(device=device, dtype=torch.float32)  # (B,C,H,W)

    weights = weights * valid  # ignore always excluded
    return {"M_gt": M_gt, "weights": weights, "valid": valid}


@torch.no_grad()
def main():
    cfg = json.loads(Path(CONFIG_PATH).read_text())

    ds = Dataset(cfg, SPLIT, LIST_PATH)
    orig_builder = GDMPBuilderGT_ORIG(
        class_ids=CLASS_IDS,
        background_id=BACKGROUND_ID,
        ignore_id=IGNORE_ID,
        fg_weight=FG_WEIGHT,
        bg_weight=BG_WEIGHT,
        boundary_ksize=BOUNDARY_KSIZE,
        eps=EPS,
    )

    n = len(ds)
    s0 = int(START_INDEX)
    s1 = min(n, s0 + int(NUM_SAMPLES))
    print(f"[INFO] Dataset size={n}, inspect [{s0}, {s1}) from list={LIST_PATH}")

    for idx in range(s0, s1):
        sample = ds[idx]
        rgb = sample["rgb"]          # (3,H,W), float
        anno = sample["anno"]        # (1,H,W) or (H,W)
        if torch.is_tensor(anno):
            if anno.ndim == 3 and anno.size(0) == 1:
                anno_hw = anno[0].long()
            elif anno.ndim == 2:
                anno_hw = anno.long()
            else:
                # fallback
                anno_hw = anno.squeeze().long()
        else:
            anno_hw = torch.from_numpy(np.array(anno)).long()

        rgb_bgr = rgb_tensor_to_bgr(rgb)
        H, W = rgb_bgr.shape[:2]

        # build gts
        anno_bhw = anno_hw.unsqueeze(0)  # (1,H,W)

        gt_orig = orig_builder(anno_bhw)
        M_orig = gt_orig["M_gt"][0].detach().cpu().numpy()  # (C,H,W)
        gt_orig_max = M_orig.max(axis=0)                    # (H,W)

        gt_ref = build_pgdp_gt_refined_from_edt(anno_bhw)
        M_ref = gt_ref["M_gt"][0].detach().cpu().numpy()
        gt_ref_max = M_ref.max(axis=0)

        # diff
        diff = (gt_ref_max - gt_orig_max).astype(np.float32)

        # viz normalization (GT itself is 0~1, diff is -1~1)
        gt_orig_gray = to_gray_png(np.clip(gt_orig_max, 0, 1))
        gt_ref_gray  = to_gray_png(np.clip(gt_ref_max, 0, 1))

        # diff heat: show positive refinement stronger
        diff_pos01 = robust_norm01(np.clip(diff, 0.0, None), p_low=0.0, p_high=99.0)
        diff_heat = to_heatmap(diff_pos01)

        # overlays
        heat_orig = to_heatmap(robust_norm01(gt_orig_max, p_low=1.0, p_high=99.0))
        heat_ref  = to_heatmap(robust_norm01(gt_ref_max,  p_low=1.0, p_high=99.0))
        over_orig = overlay(rgb_bgr, heat_orig, OVERLAY_ALPHA)
        over_ref  = overlay(rgb_bgr, heat_ref,  OVERLAY_ALPHA)

        # anno color
        anno_np = anno_hw.detach().cpu().numpy().astype(np.int32)
        anno_col = colorize_anno(anno_np)

        # save
        out = OUT_DIR / f"idx_{idx:06d}"
        out.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out / "rgb.png"), rgb_bgr)
        cv2.imwrite(str(out / "anno_color.png"), anno_col)

        cv2.imwrite(str(out / "pgdp_gt_orig_gray.png"), gt_orig_gray)
        cv2.imwrite(str(out / "pgdp_gt_refined_gray.png"), gt_ref_gray)

        cv2.imwrite(str(out / "pgdp_gt_orig_overlay.png"), over_orig)
        cv2.imwrite(str(out / "pgdp_gt_refined_overlay.png"), over_ref)

        cv2.imwrite(str(out / "diff_ref_minus_orig_heat.png"), diff_heat)

        # also dump quick stats
        print(
            f"[{idx}] orig_mean={gt_orig_max.mean():.4f} ref_mean={gt_ref_max.mean():.4f} "
            f"diff_pos_mean={np.clip(diff,0,None).mean():.5f}"
        )

    print(f"[DONE] Saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()