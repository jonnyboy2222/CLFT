#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.dataset import Dataset
from clft_ctca_fusion.clft import CLFT
from tools.clft_cwt_patch_gt import build_patch_gt_soft
from tools.clft_cwt_utils import compute_slot_class_score, match_slots_k2, reorder_slots


CONFIG_PATH = "/home/john/dev_ws/CLFT/config.json"
CKPT_PATH   = "/home/john/dev_ws/CLFT/logs/clft_fusioncheckpoint_48.pth"
DEMO_LIST   = "/home/john/dev_ws/CLFT/waymo_dataset/visual_run_demo.txt"

SPLIT = "val"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_ROOT = Path("./cwt_viz_out_reordered")

COLORMAP = cv2.COLORMAP_TURBO
OVERLAY_ALPHA = 0.35
HEAT_GAIN = 1.3
TOPK_RATIO = 0.10  # top 10% patches


def minmax01(x, eps=1e-6):
    return (x - x.min()) / (x.max() - x.min() + eps)


def rgb_tensor_to_bgr(rgb):
    img = rgb.permute(1, 2, 0).detach().cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def anno_to_color(anno):
    """Simple debug palette: bg=black, car=green, person=red, ignore=gray"""
    a = anno.detach().cpu().numpy().astype(np.int64)
    h, w = a.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[a == 0] = (0, 0, 0)
    out[a == 1] = (0, 255, 0)
    out[a == 2] = (0, 0, 255)
    out[a == 255] = (128, 128, 128)
    return out


def to_heatmap(arr01, H, W):
    arr = cv2.resize(arr01, (W, H), interpolation=cv2.INTER_CUBIC)
    arr = np.clip(arr, 0, 1)
    heat = cv2.applyColorMap((arr * 255).astype(np.uint8), COLORMAP)
    heat = np.clip(heat.astype(np.float32) * HEAT_GAIN, 0, 255).astype(np.uint8)
    return heat


def overlay(base, heat, alpha=0.35):
    return cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)


def infer_condition(path):
    p = path.replace("\\", "/").split("/")
    if "labeled" in p:
        i = p.index("labeled")
        return f"{p[i+1]}_{p[i+2]}"
    return "unknown"


def build_model(cfg):
    r = cfg["Dataset"]["transforms"]["resize"]
    return CLFT(
        RGB_tensor_size=(3, r, r),
        XYZ_tensor_size=(3, r, r),
        patch_size=cfg["CLFT"]["patch_size"],
        emb_dim=cfg["CLFT"]["emb_dim"],
        resample_dim=cfg["CLFT"]["resample_dim"],
        read=cfg["CLFT"]["read"],
        hooks=cfg["CLFT"]["hooks"],
        reassemble_s=cfg["CLFT"]["reassembles"],
        nclasses=len(cfg["Dataset"]["classes"]),
        type=cfg["CLFT"]["type"],
        model_timm=cfg["CLFT"]["model_timm"],
    ).to(DEVICE)


def build_patch_mask_from_lidar(lidar_bchw, patch_size):
    """
    lidar_bchw: (B, C, H, W)
    return: (B, Np) float mask in {0,1}
    """
    occ = (lidar_bchw.abs().sum(dim=1) > 0).float()  # (B,H,W)
    B, H, W = occ.shape
    assert H % patch_size == 0 and W % patch_size == 0
    Hp, Wp = H // patch_size, W // patch_size

    occ_patch = occ.view(B, Hp, patch_size, Wp, patch_size)
    occ_patch = occ_patch.permute(0, 1, 3, 2, 4).contiguous()  # (B,Hp,Wp,ps,ps)
    occ_patch = (occ_patch.amax(dim=(-1, -2)) > 0).float()     # (B,Hp,Wp)
    return occ_patch.view(B, Hp * Wp)


def save_single_map(base, arr2d, out_png, alpha=OVERLAY_ALPHA):
    H, W = base.shape[:2]
    heat = to_heatmap(minmax01(arr2d), H, W)
    over = overlay(base, heat, alpha)
    cv2.imwrite(str(out_png), over)


def save_binary_map(base, bin2d, out_png, color=(0, 255, 255), alpha=0.45):
    H, W = base.shape[:2]
    mask = cv2.resize(bin2d.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    color_img[mask > 0] = color
    out = cv2.addWeighted(base, 1.0, color_img, alpha, 0)
    cv2.imwrite(str(out_png), out)


def save_assign_map(assign2d, H, W, out_png, K=2):
    assign_vis = (assign2d.astype(np.uint8) * (255 // max(K - 1, 1))).astype(np.uint8)
    assign_vis = cv2.resize(assign_vis, (W, H), interpolation=cv2.INTER_NEAREST)
    assign_vis = cv2.applyColorMap(assign_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_png), assign_vis)


def save_cwt_maps(base, w_bkn, Hp, Wp, out_dir, prefix, patch_mask=None):
    """
    w_bkn: (B, K, Np)
    patch_mask: (B, Np) or None
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    w0 = w_bkn[0].detach().cpu().numpy()  # (K, Np)
    K, Np = w0.shape
    H, W = base.shape[:2]

    pm = None
    if patch_mask is not None:
        pm = patch_mask[0].detach().cpu().numpy().reshape(Hp, Wp)
        save_binary_map(base, pm, out_dir / f"{prefix}_patchmask.png", color=(255, 255, 255), alpha=0.35)

    for k in range(K):
        a2d = w0[k].reshape(Hp, Wp)
        save_single_map(base, a2d, out_dir / f"{prefix}_slot{k}.png")

        # top-k patch support map (helps when peak looks too narrow)
        thr = np.quantile(a2d.reshape(-1), max(0.0, 1.0 - TOPK_RATIO))
        topk_bin = (a2d >= thr).astype(np.uint8)
        save_binary_map(base, topk_bin, out_dir / f"{prefix}_slot{k}_topk{int(TOPK_RATIO*100)}.png")

        if pm is not None:
            a2d_masked = a2d * pm
            save_single_map(base, a2d_masked, out_dir / f"{prefix}_slot{k}_maskaware.png")

            thr_m = np.quantile(a2d_masked.reshape(-1), max(0.0, 1.0 - TOPK_RATIO))
            topk_bin_m = (a2d_masked >= thr_m).astype(np.uint8)
            save_binary_map(base, topk_bin_m, out_dir / f"{prefix}_slot{k}_maskaware_topk{int(TOPK_RATIO*100)}.png", color=(255, 128, 0))

    assign = np.argmax(w0, axis=0).reshape(Hp, Wp).astype(np.uint8)
    save_assign_map(assign, H, W, out_dir / f"{prefix}_argmax.png", K=K)

    if pm is not None:
        w0_masked = w0 * pm.reshape(1, -1)
        assign_m = np.argmax(w0_masked, axis=0).reshape(Hp, Wp).astype(np.uint8)
        save_assign_map(assign_m, H, W, out_dir / f"{prefix}_maskaware_argmax.png", K=K)


def reorder_stage(extras, patch_gt, stage="s2"):
    """Mirror trainer logic for visualization."""
    key_w_cam = f"w_cam_{stage}"
    key_w_xyz = f"w_xyz_{stage}"
    key_t_cam = f"tok_cam_{stage}"
    key_t_xyz = f"tok_xyz_{stage}"

    out = {}

    if key_w_cam in extras and key_t_cam in extras:
        slot_score_raw_cam = compute_slot_class_score(extras[key_w_cam], patch_gt)
        perm_idx_cam = match_slots_k2(slot_score_raw_cam)
        tok_cam, w_cam = reorder_slots(extras[key_t_cam], extras[key_w_cam], perm_idx_cam)
        out["perm_idx_cam"] = perm_idx_cam
        out["tok_cam"] = tok_cam
        out["w_cam"] = w_cam
        out["slot_score_cam"] = compute_slot_class_score(w_cam, patch_gt)

    if key_w_xyz in extras and key_t_xyz in extras:
        slot_score_raw_xyz = compute_slot_class_score(extras[key_w_xyz], patch_gt)
        perm_idx_xyz = match_slots_k2(slot_score_raw_xyz)
        tok_xyz, w_xyz = reorder_slots(extras[key_t_xyz], extras[key_w_xyz], perm_idx_xyz)
        out["perm_idx_xyz"] = perm_idx_xyz
        out["tok_xyz"] = tok_xyz
        out["w_xyz"] = w_xyz
        out["slot_score_xyz"] = compute_slot_class_score(w_xyz, patch_gt)

    return out


@torch.no_grad()
def main():
    cfg = json.loads(Path(CONFIG_PATH).read_text())
    patch_size = cfg["CLFT"]["patch_size"]

    model = build_model(cfg)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    demo_lines = [x for x in Path(DEMO_LIST).read_text().splitlines() if x.strip()]

    for rel in demo_lines:
        cond = infer_condition(rel)
        stem = Path(rel).stem
        out_dir = OUT_ROOT / cond / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp = out_dir / "_tmp.txt"
        tmp.write_text(rel + "\n")

        ds = Dataset(cfg, SPLIT, str(tmp))
        s = ds[0]

        rgb = s["rgb"].unsqueeze(0).to(DEVICE)
        lidar = s["lidar"].unsqueeze(0).to(DEVICE)
        anno = s["anno"]
        if anno.ndim == 3 and anno.size(0) == 1:
            anno = anno.squeeze(0)
        anno_b = anno.unsqueeze(0).to(DEVICE)

        _, _, extras = model(rgb, lidar, modal="cross_fusion", return_extras=True)

        base = rgb_tensor_to_bgr(rgb[0])
        cv2.imwrite(str(out_dir / "rgb.png"), base)
        cv2.imwrite(str(out_dir / "anno.png"), anno_to_color(anno))

        Hp, Wp = extras["HpWp"]
        patch_gt, cls_mask = build_patch_gt_soft(
            gt=anno_b,
            patch_size=patch_size,
            class_ids=(2, 1),
            ignore_index=255,
        )
        patch_mask = build_patch_mask_from_lidar(lidar, patch_size)

        reordered_s2 = reorder_stage(extras, patch_gt, stage="s2")

        meta = {
            "rel": rel,
            "cls_mask": cls_mask[0].detach().cpu().tolist(),
        }

        if "perm_idx_cam" in reordered_s2:
            meta["perm_idx_cam_s2"] = reordered_s2["perm_idx_cam"][0].detach().cpu().tolist()
            meta["slot_score_cam_s2"] = reordered_s2["slot_score_cam"][0].detach().cpu().tolist()
            save_cwt_maps(
                base,
                reordered_s2["w_cam"],
                Hp, Wp,
                out_dir,
                "cam_s2_reordered",
                patch_mask=None,
            )

        # if "perm_idx_xyz" in reordered_s2:
        #     meta["perm_idx_xyz_s2"] = reordered_s2["perm_idx_xyz"][0].detach().cpu().tolist()
        #     meta["slot_score_xyz_s2"] = reordered_s2["slot_score_xyz"][0].detach().cpu().tolist()
        #     save_cwt_maps(
        #         base,
        #         reordered_s2["w_xyz"],
        #         Hp, Wp,
        #         out_dir,
        #         "xyz_s2_reordered",
        #         patch_mask=patch_mask,
        #     )

        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        tmp.unlink(missing_ok=True)

    print(f"[DONE] Reordered CWT visualization saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
