#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gate_viz.py

- Visualize TWO kinds of gates:
  (A) Fusion gate: per-stage Fusion.gate (S0~S3)
  (B) CTCA token-update gate: stage0/stage2 token gate (g0/g2)  (requires model.last_tokgate_map)

- Save outputs into separate folders:
  gate_viz_out/
    fusion/<cond>/gate_S{0..3}.png
    fusion/fusion_gate_summary.csv

    ctca/<cond>/tokgate_S0.png
    ctca/<cond>/tokgate_S2.png
    ctca/ctca_gate_summary.csv
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
CKPT_PATH   = "/home/john/dev_ws/CLFT/logs/clft_ctca_v4_fusioncheckpoint_89_9210_6911.pth"
DEMO_LIST_PATH = "/home/john/dev_ws/CLFT/waymo_dataset/visual_run_demo.txt"
SPLIT = "val"

OUT_ROOT = Path("./gate_viz_out")

# overlay strength (RGB visibility)
FUSION_OVERLAY_ALPHA = 0.3
CTCA_OVERLAY_ALPHA   = 0.3

# heatmap intensity scaling
HEAT_GAIN_FUSION = 1.2
HEAT_GAIN_CTCA   = 1.2
# ==========================================================


def build_model(cfg: dict, device: str):
    resize = cfg["Dataset"]["transforms"]["resize"]
    model = CLFT(
        RGB_tensor_size=(3, resize, resize),
        XYZ_tensor_size=(3, resize, resize),
        patch_size=cfg["CLFT"]["patch_size"],
        emb_dim=cfg["CLFT"]["emb_dim"],
        resample_dim=cfg["CLFT"]["resample_dim"],
        read=cfg["CLFT"]["read"],
        hooks=cfg["CLFT"]["hooks"],
        reassemble_s=cfg["CLFT"]["reassembles"],
        nclasses=len(cfg["Dataset"]["classes"]),
        type=cfg["CLFT"]["type"],
        model_timm=cfg["CLFT"]["model_timm"],
    ).to(device)
    return model


def load_ckpt(model, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)


def read_demo_lines(p: str):
    return [ln.strip() for ln in Path(p).read_text().splitlines() if ln.strip()]


def infer_condition_name(rel_path: str) -> str:
    toks = rel_path.replace("\\", "/").split("/")
    if "labeled" in toks:
        i = toks.index("labeled")
        return f"{toks[i+1]}_{toks[i+2]}"  # day_rain etc.
    return "unknown"


def tensor_rgb_to_bgr_uint8(rgb: torch.Tensor) -> np.ndarray:
    x = rgb.detach().cpu().permute(1, 2, 0).numpy()
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def tensor_1ch_to_heatmap_bgr(t1: torch.Tensor, H: int, W: int, gain: float) -> np.ndarray:
    """
    t1: (1,h,w) or (h,w), expected [0,1]
    return: heatmap BGR uint8
    """
    g = t1.detach().cpu()
    if g.ndim == 3:
        g = g.squeeze(0)
    g = g.numpy()

    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    g = np.clip(g, 0.0, 1.0)

    heat = (g * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = np.clip(heat.astype(np.float32) * float(gain), 0, 255).astype(np.uint8)
    return heat


def overlay_heatmap(base_bgr: np.ndarray, heat_bgr: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))
    return cv2.addWeighted(base_bgr, 1.0 - alpha, heat_bgr, alpha, 0)


@torch.no_grad()
def run_one_sample(model, cfg, rel_path: str, device: str,
                   fusion_cond_dir: Path, ctca_cond_dir: Path,
                   fusion_rows: list, ctca_rows: list):
    # Dataset은 list 파일을 받으므로 임시 list 생성
    tmp_list = (fusion_cond_dir / "_tmp_list.txt")
    tmp_list.write_text(rel_path + "\n")

    ds = Dataset(config=cfg, split=SPLIT, path=str(tmp_list))
    sample = ds[0]

    rgb = sample["rgb"].unsqueeze(0).to(device)     # (1,3,H,W)
    lidar = sample["lidar"].unsqueeze(0).to(device) # (1,3,H,W)

    _ = model(rgb, lidar, modal="cross_fusion")

    base = tensor_rgb_to_bgr_uint8(rgb[0])
    H, W = base.shape[:2]
    cond = infer_condition_name(rel_path)

    # ======================================================
    # (A) Fusion gate: per stage S0~S3
    # ======================================================
    for si, fus in enumerate(model.fusions):
        g_map = getattr(fus, "last_gate_map", None)
        if g_map is None:
            print(f"[WARN] Fusion S{si}: no last_gate_map. "
                  f"Add `self.last_gate_map = g.detach()` in Fusion.forward() (modal=='cross_fusion').")
            continue

        g0 = g_map[0]  # (1,h,w)
        g_mean = float(g0.mean().item())

        last_log = getattr(fus, "last_log", {}) or {}
        scale_mean = float(last_log.get("scale", float("nan")))

        fusion_rows.append({
            "condition": cond,
            "stage": f"S{si}",
            "gate_mean": f"{g_mean:.3f}",
            "scale_mean": f"{scale_mean:.3f}",
        })

        heat = tensor_1ch_to_heatmap_bgr(g0, H, W, gain=HEAT_GAIN_FUSION)
        over = overlay_heatmap(base, heat, alpha=FUSION_OVERLAY_ALPHA)
        cv2.imwrite(str(fusion_cond_dir / f"gate_S{si}.png"), over)

    # ======================================================
    # (B) CTCA token-update gate: stage0/stage2 (requires map)
    # ======================================================
    tok_map = getattr(model, "last_tokgate_map", None)
    tok_log = getattr(model, "last_tokgate", None)

    if tok_map is None or tok_log is None:
        print("[WARN] model has no last_tokgate_map/last_tokgate. "
              "Please store token gate maps in clft.py (see patch snippet).")
    else:
        for key in ["S0", "S2"]:
            g2d = tok_map.get(key, None)  # (B,1,Hp,Wp)
            if g2d is None:
                print(f"[WARN] CTCA tok gate {key}: map is None (not produced this forward?)")
                continue

            # map -> overlay (upsample internally)
            g2d0 = g2d[0]  # (1,Hp,Wp)
            g_mean = float(tok_log[key]["g_mean"]) if tok_log.get(key) else float("nan")
            g_std  = float(tok_log[key]["g_std"])  if tok_log.get(key) else float("nan")

            ctca_rows.append({
                "condition": cond,
                "stage": key,   # "S0" or "S2"
                "tok_gate_mean": f"{g_mean:.3f}",
                "tok_gate_std":  f"{g_std:.3f}",
            })

            heat = tensor_1ch_to_heatmap_bgr(g2d0, H, W, gain=HEAT_GAIN_CTCA)
            over = overlay_heatmap(base, heat, alpha=CTCA_OVERLAY_ALPHA)
            cv2.imwrite(str(ctca_cond_dir / f"tokgate_{key}.png"), over)

    # cleanup
    tmp_list.unlink(missing_ok=True)


def write_csv(path: Path, rows: list, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    # output folders
    fusion_root = OUT_ROOT / "fusion"
    ctca_root   = OUT_ROOT / "ctca"
    fusion_root.mkdir(parents=True, exist_ok=True)
    ctca_root.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(CONFIG_PATH).read_text())
    device = cfg["General"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu"

    model = build_model(cfg, device)
    load_ckpt(model, CKPT_PATH, device)
    model.eval()

    demo_lines = read_demo_lines(DEMO_LIST_PATH)

    fusion_rows = []
    ctca_rows   = []

    print(f"[INFO] device={device} split={SPLIT}")
    print(f"[INFO] out_root={OUT_ROOT.resolve()}")
    print(f"[INFO] demo_list={DEMO_LIST_PATH} (num={len(demo_lines)})")

    for rel in demo_lines:
        cond = infer_condition_name(rel)

        fusion_cond_dir = fusion_root / cond
        ctca_cond_dir   = ctca_root / cond
        fusion_cond_dir.mkdir(parents=True, exist_ok=True)
        ctca_cond_dir.mkdir(parents=True, exist_ok=True)

        print(f"[RUN] {cond}")
        run_one_sample(
            model=model, cfg=cfg, rel_path=rel, device=device,
            fusion_cond_dir=fusion_cond_dir,
            ctca_cond_dir=ctca_cond_dir,
            fusion_rows=fusion_rows,
            ctca_rows=ctca_rows
        )

    # CSVs (separate)
    write_csv(
        fusion_root / "fusion_gate_summary.csv",
        fusion_rows,
        fieldnames=["condition", "stage", "gate_mean", "scale_mean"]
    )
    write_csv(
        ctca_root / "ctca_tokgate_summary.csv",
        ctca_rows,
        fieldnames=["condition", "stage", "tok_gate_mean", "tok_gate_std"]
    )

    print(f"[OK] saved fusion CSV: {(fusion_root / 'fusion_gate_summary.csv').resolve()}")
    print(f"[OK] saved ctca  CSV: {(ctca_root / 'ctca_tokgate_summary.csv').resolve()}")
    print("[DONE]")


if __name__ == "__main__":
    main()
