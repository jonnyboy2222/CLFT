#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.dataset import Dataset
from clft.clft import CLFT


CONFIG_PATH = "/home/john/dev_ws/CLFT/config.json"
CKPT_PATH   = "/home/john/dev_ws/CLFT/logs/clft_ctca_fusion_v1/clft_ctca_v1_fusioncheckpoint_87_921_689.pth"
DEMO_LIST   = "/home/john/dev_ws/CLFT/waymo_dataset/visual_run_demo.txt"

SPLIT = "val"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_ROOT = Path("./cwt_viz_out")

COLORMAP = cv2.COLORMAP_TURBO
OVERLAY_ALPHA = 0.35
HEAT_GAIN = 1.3


def minmax01(x, eps=1e-6):
    return (x - x.min()) / (x.max() - x.min() + eps)


def rgb_tensor_to_bgr(rgb):
    img = rgb.permute(1, 2, 0).cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


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


def save_cwt_maps(base, w_bkn, Hp, Wp, out_dir, prefix):
    """
    w_bkn: (B, K, Np)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    w0 = w_bkn[0].detach().cpu().numpy()  # (K, Np)
    K, Np = w0.shape
    H, W = base.shape[:2]

    for k in range(K):
        a2d = w0[k].reshape(Hp, Wp)
        heat = to_heatmap(minmax01(a2d), H, W)
        over = overlay(base, heat, OVERLAY_ALPHA)
        cv2.imwrite(str(out_dir / f"{prefix}_slot{k}.png"), over)

    # argmax assignment map
    assign = np.argmax(w0, axis=0).reshape(Hp, Wp).astype(np.uint8)
    assign_vis = (assign * (255 // max(K - 1, 1))).astype(np.uint8)
    assign_vis = cv2.resize(assign_vis, (W, H), interpolation=cv2.INTER_NEAREST)
    assign_vis = cv2.applyColorMap(assign_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_dir / f"{prefix}_argmax.png"), assign_vis)


@torch.no_grad()
def main():
    cfg = json.loads(Path(CONFIG_PATH).read_text())
    model = build_model(cfg)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    demo_lines = Path(DEMO_LIST).read_text().splitlines()

    for rel in demo_lines:
        cond = infer_condition(rel)
        out_dir = OUT_ROOT / cond
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp = out_dir / "_tmp.txt"
        tmp.write_text(rel + "\n")

        ds = Dataset(cfg, SPLIT, str(tmp))
        s = ds[0]

        rgb = s["rgb"].unsqueeze(0).to(DEVICE)
        lidar = s["lidar"].unsqueeze(0).to(DEVICE)

        _, _, extras = model(rgb, lidar, modal="cross_fusion", return_extras=True)

        base = rgb_tensor_to_bgr(rgb[0])
        Hp, Wp = extras["HpWp"]

        save_cwt_maps(base, extras["w_cam_s2"], Hp, Wp, out_dir, "cam_s2")
        save_cwt_maps(base, extras["w_xyz_s2"], Hp, Wp, out_dir, "xyz_s2")
        save_cwt_maps(base, extras["w_cam_s0"], Hp, Wp, out_dir, "cam_s0")
        save_cwt_maps(base, extras["w_xyz_s0"], Hp, Wp, out_dir, "xyz_s0")

        tmp.unlink(missing_ok=True)

    print("[DONE] CWT visualization saved.")


if __name__ == "__main__":
    main()