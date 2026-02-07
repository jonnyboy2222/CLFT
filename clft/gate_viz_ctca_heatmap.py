#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gate_viz_ctca_heatmap.py

- Fusion gate heatmap + CSV (기존 유지)
- CTCA delta_norm heatmap (stage별 저장)
- CTCA gate 완전 제거
"""

import json
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.dataset import Dataset
from clft.clft import CLFT


# ==========================================================
# USER CONFIG
# ==========================================================
CONFIG_PATH = "/home/john/dev_ws/CLFT/config.json"
CKPT_PATH   = "/home/john/dev_ws/CLFT/logs/clft_ctca_v5_fusioncheckpoint_90_9245_6807.pth"
DEMO_LIST   = "/home/john/dev_ws/CLFT/waymo_dataset/visual_run_demo.txt"

SPLIT = "val"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

OUT_ROOT = Path("./gate_viz_out_ctca_heatmap")

CTCA_STAGES = [0, 2]      # CTCA 적용 stage
FUSION_ALPHA = 0.30
DELTA_ALPHA  = 0.30
HEAT_GAIN    = 1.3

COLORMAP = cv2.COLORMAP_TURBO
# ==========================================================


def minmax01(x, eps=1e-6):
    return (x - x.min()) / (x.max() - x.min() + eps)


def overlay(base, heat, alpha):
    return cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)


def to_heatmap(arr01, H, W):
    arr = cv2.resize(arr01, (W, H))
    arr = np.clip(arr, 0, 1)
    heat = cv2.applyColorMap((arr * 255).astype(np.uint8), COLORMAP)
    heat = np.clip(heat.astype(np.float32) * HEAT_GAIN, 0, 255).astype(np.uint8)
    return heat


def rgb_tensor_to_bgr(rgb):
    img = rgb.permute(1, 2, 0).cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


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


@torch.no_grad()
def main():
    cfg = json.loads(Path(CONFIG_PATH).read_text())

    model = build_model(cfg)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    demo_lines = Path(DEMO_LIST).read_text().splitlines()

    fusion_csv = []
    ctca_csv = []

    for rel in demo_lines:
        cond = infer_condition(rel)

        out_fusion = OUT_ROOT / "fusion" / cond
        out_ctca   = OUT_ROOT / "ctca" / cond
        out_fusion.mkdir(parents=True, exist_ok=True)
        out_ctca.mkdir(parents=True, exist_ok=True)

        tmp = out_fusion / "_tmp.txt"
        tmp.write_text(rel + "\n")

        ds = Dataset(cfg, SPLIT, str(tmp))
        s = ds[0]

        rgb = s["rgb"].unsqueeze(0).to(DEVICE)
        lidar = s["lidar"].unsqueeze(0).to(DEVICE)

        _ = model(rgb, lidar, modal="cross_fusion")

        base = rgb_tensor_to_bgr(rgb[0])
        H, W = base.shape[:2]

        # ------------------------
        # Fusion gate (기존 유지)
        # ------------------------
        for si, fus in enumerate(model.fusions):
            gmap = getattr(fus, "last_gate_map", None)
            if gmap is None:
                continue

            g = gmap[0, 0].cpu().numpy()
            heat = to_heatmap(minmax01(g), H, W)
            over = overlay(base, heat, FUSION_ALPHA)
            last_log = getattr(fus, "last_log", {})

            cv2.imwrite(str(out_fusion / f"gate_S{si}.png"), over)

            fusion_csv.append({
                "condition": cond,
                "stage": f"S{si}",
                "g_mean":    last_log.get("g", float("nan")),
                "scale_mean": last_log.get("scale", float("nan")),
            })

        # ------------------------
        # CTCA delta_norm (stage별)
        # ------------------------
        for si in CTCA_STAGES:
            dn = getattr(model, f"last_delta_norm{si}", None)
            hw = getattr(model, f"last_delta_hw{si}", None)

            if dn is None or hw is None:
                continue

            Hp, Wp = hw
            dn0 = dn[0].cpu().numpy()

            if dn0.size != Hp * Wp:
                continue

            dn2d = dn0.reshape(Hp, Wp)
            heat = to_heatmap(minmax01(dn2d), H, W)
            over = overlay(base, heat, DELTA_ALPHA)

            cv2.imwrite(str(out_ctca / f"delta_S{si}.png"), over)

            ctca_csv.append({
                "condition": cond,
                "stage": f"S{si}",
                "delta_mean": float(dn0.mean()),
                "delta_std":  float(dn0.std()),
                "delta_max":  float(dn0.max()),
            })

        tmp.unlink(missing_ok=True)

    # CSV 저장
    with open(OUT_ROOT / "fusion_gate_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fusion_csv[0].keys())
        w.writeheader()
        w.writerows(fusion_csv)

    with open(OUT_ROOT / "ctca_delta_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ctca_csv[0].keys())
        w.writeheader()
        w.writerows(ctca_csv)

    print("[DONE] visualization & CSV saved.")


if __name__ == "__main__":
    main()
