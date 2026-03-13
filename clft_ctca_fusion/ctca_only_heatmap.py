#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ctca_delta_heatmap_only.py

- Visualize CTCA delta_norm only (NO fusion gate)
- Uses:
    model.last_delta_norm0, model.last_delta_hw0
    model.last_delta_norm2, model.last_delta_hw2
- Saves:
    out_root/ctca/<cond>/delta_S0.png
    out_root/ctca/<cond>/delta_S2.png
    out_root/ctca_delta_summary.csv
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
CKPT_PATH   = "/home/john/dev_ws/CLFT/logs/clft_fusioncheckpoint_83.pth"
DEMO_LIST   = "/home/john/dev_ws/CLFT/waymo_dataset/visual_run_demo.txt"

SPLIT  = "val"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

OUT_ROOT = Path("./ctca_only_heatmap")

CTCA_STAGES = [0, 2]   # stage0, stage2
OVERLAY_ALPHA = 0.30
HEAT_GAIN = 1.30
COLORMAP = cv2.COLORMAP_TURBO
# ==========================================================


def infer_condition(path: str) -> str:
    p = path.replace("\\", "/").split("/")
    if "labeled" in p:
        i = p.index("labeled")
        return f"{p[i+1]}_{p[i+2]}"
    return "unknown"


def rgb_tensor_to_bgr(rgb: torch.Tensor) -> np.ndarray:
    # rgb: (3,H,W) float [0,1]
    img = rgb.permute(1, 2, 0).detach().cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def minmax01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)


def to_heatmap(arr01_hw: np.ndarray, H: int, W: int) -> np.ndarray:
    arr = cv2.resize(arr01_hw, (W, H), interpolation=cv2.INTER_LINEAR)
    arr = np.clip(arr, 0, 1)
    heat = cv2.applyColorMap((arr * 255).astype(np.uint8), COLORMAP)
    heat = np.clip(heat.astype(np.float32) * HEAT_GAIN, 0, 255).astype(np.uint8)
    return heat


def overlay(base_bgr: np.ndarray, heat_bgr: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))
    return cv2.addWeighted(base_bgr, 1 - alpha, heat_bgr, alpha, 0)


def build_model(cfg: dict) -> CLFT:
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

    demo_lines = [ln.strip() for ln in Path(DEMO_LIST).read_text().splitlines() if ln.strip()]

    out_ctca = OUT_ROOT / "ctca"
    out_ctca.mkdir(parents=True, exist_ok=True)

    rows = []

    for rel in demo_lines:
        cond = infer_condition(rel)
        out_dir = out_ctca / cond
        out_dir.mkdir(parents=True, exist_ok=True)

        # Dataset은 list 파일을 받으므로 임시 파일 생성
        tmp_list = out_dir / "_tmp_list.txt"
        tmp_list.write_text(rel + "\n")

        ds = Dataset(cfg, SPLIT, str(tmp_list))
        s = ds[0]

        rgb = s["rgb"].unsqueeze(0).to(DEVICE)
        lidar = s["lidar"].unsqueeze(0).to(DEVICE)

        _ = model(rgb, lidar, modal="cross_fusion")

        base = rgb_tensor_to_bgr(rgb[0])
        H, W = base.shape[:2]

        # ---- CTCA delta_norm only ----
        for si in CTCA_STAGES:
            dn = getattr(model, f"last_delta_norm{si}", None)  # (B,N)
            hw = getattr(model, f"last_delta_hw{si}", None)    # (Hp,Wp)

            if dn is None or hw is None:
                continue

            Hp, Wp = hw
            dn0 = dn[0].detach().cpu().numpy()  # (N,)

            # safety
            if dn0.size != Hp * Wp:
                print(f"[WARN] delta_norm{si} size mismatch: N={dn0.size}, Hp*Wp={Hp*Wp}")
                continue

            dn2d = dn0.reshape(Hp, Wp)  # (Hp,Wp)
            heat = to_heatmap(minmax01(dn2d), H, W)
            over = overlay(base, heat, OVERLAY_ALPHA)

            cv2.imwrite(str(out_dir / f"delta_S{si}.png"), over)

            rows.append({
                "condition": cond,
                "stage": f"S{si}",
                "delta_mean": float(dn0.mean()),
                "delta_std":  float(dn0.std()),
                "delta_max":  float(dn0.max()),
            })

        tmp_list.unlink(missing_ok=True)

    # summary CSV
    csv_path = OUT_ROOT / "ctca_delta_summary.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"[DONE] CTCA delta heatmaps saved to: {OUT_ROOT.resolve()}")
    if rows:
        print(f"[DONE] CSV saved: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
