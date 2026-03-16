#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.dataset import Dataset
from clft.clft import CLFT

# PGDP GT builder (trainer.py에서 쓰는 것과 동일)
from tools.clft_gdmp_gt import GDMPBuilderGT

"""
gs_viz.py

- SDUP u_map (LEFT, gray)
- GDMP pred p0 (RIGHT, heatmap)
- GDM GT map (RIGHT, gray)

Saves per condition sample:
    out_root/gs_maps/<cond>/sdup_u_gray.png
    out_root/gs_maps/<cond>/gdmp_pred_heat.png
    out_root/gs_maps/<cond>/gdmp_pred_overlay.png
    out_root/gs_maps/<cond>/gdm_gt_gray.png
"""


# ==========================================================
# USER CONFIG (ctca_only_heatmap.py랑 동일 스타일)
# ==========================================================
CONFIG_PATH = "/home/john/dev_ws/CLFT/config.json"
CKPT_PATH   = "/home/john/dev_ws/CLFT/logs/clft_fusioncheckpoint_12.pth"
DEMO_LIST   = "/home/john/dev_ws/CLFT/waymo_dataset/visual_run_demo.txt"

SPLIT  = "val"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

OUT_ROOT = Path("./clft_gs_heatmap")

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


def to_gray_png(arr01_hw: np.ndarray, H: int, W: int) -> np.ndarray:
    """0~1 map을 gray uint8 이미지로 저장"""
    arr = cv2.resize(arr01_hw, (W, H), interpolation=cv2.INTER_LINEAR)
    arr = np.clip(arr, 0, 1)
    gray = (arr * 255).astype(np.uint8)
    return gray


def robust_norm01(x: np.ndarray, p_low=1.0, p_high=95.0, eps=1e-6) -> np.ndarray:
    """PFIM/PGDP처럼 spike 가능할 때 percentile로 안정화 정규화"""
    x = x.astype(np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    return np.clip((x - lo) / (hi - lo + eps), 0, 1)


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


def find_first_module_with_attr(model: torch.nn.Module, attr: str):
    for _, m in model.named_modules():
        if hasattr(m, attr):
            return m
    return None


@torch.no_grad()
def main():
    cfg = json.loads(Path(CONFIG_PATH).read_text())

    model = build_model(cfg)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    pgdp_gt_builder = GDMPBuilderGT(refine=True)

    demo_lines = [ln.strip() for ln in Path(DEMO_LIST).read_text().splitlines() if ln.strip()]

    out_root = OUT_ROOT / "pfim_pgdp"
    out_root.mkdir(parents=True, exist_ok=True)

    for rel in demo_lines:
        cond = infer_condition(rel)
        out_dir = out_root / cond
        out_dir.mkdir(parents=True, exist_ok=True)

        # Dataset은 list 파일을 받으므로 임시 파일 생성 (ctca_only_heatmap.py 방식 그대로)
        tmp_list = out_dir / "_tmp_list.txt"
        tmp_list.write_text(rel + "\n")

        ds = Dataset(cfg, SPLIT, str(tmp_list))
        s = ds[0]

        rgb   = s["rgb"].unsqueeze(0).to(DEVICE)
        lidar = s["lidar"].unsqueeze(0).to(DEVICE)
        anno  = s["anno"].unsqueeze(0).to(DEVICE).squeeze(1)  # (B,H,W) 형태 맞추기

        # forward 한번으로 PFIM/PGDP의 last_*가 채워지게 (PATCH 1/2가 들어가 있어야 함)
        _ = model(rgb, lidar, modal="cross_fusion")

        base = rgb_tensor_to_bgr(rgb[0])
        H, W = base.shape[:2]

        # ---- SDUP (LEFT, gray) ----
        sdup_mod = getattr(model, "sdup", None)
        if sdup_mod is None:
            sdup_mod = find_first_module_with_attr(model, "last_info_map")

        if sdup_mod is not None and hasattr(sdup_mod, "last_info_map") and sdup_mod.last_info_map is not None:
            u = sdup_mod.last_info_map[0]   # (1,Hs,Ws)
            u = u.detach().float().cpu().numpy()
            if u.ndim == 3:
                u = u.mean(axis=0)
            u01 = robust_norm01(u, p_low=1.0, p_high=95.0)
            gray = to_gray_png(u01, H, W)
            cv2.imwrite(str(out_dir / "sdup_u_gray.png"), gray)

            # optional: gain map도 저장 가능
            if hasattr(sdup_mod, "last_gain_map") and sdup_mod.last_gain_map is not None:
                g = sdup_mod.last_gain_map[0].detach().float().cpu().numpy()
                if g.ndim == 3:
                    g = g.mean(axis=0)
                g01 = robust_norm01(g, p_low=1.0, p_high=95.0)
                ggray = to_gray_png(g01, H, W)
                cv2.imwrite(str(out_dir / "sdup_gain_gray.png"), ggray)
        else:
            print(f"[WARN] SDUP last_info_map not found for {cond}")

        # ---- GDMP pred (RIGHT, heatmap) ----
        gdmp_mod = getattr(model, "gdmp", None)
        if gdmp_mod is None:
            gdmp_mod = find_first_module_with_attr(model, "last_pgdp_p0")

        if gdmp_mod is not None and hasattr(gdmp_mod, "last_pgdp_p0") and gdmp_mod.last_pgdp_p0 is not None:
            p0 = gdmp_mod.last_pgdp_p0[0].detach().float().cpu().numpy()  # (C,Hs,Ws)
            p0_max = p0.max(axis=0)
            p0_01 = robust_norm01(p0_max, p_low=1.0, p_high=99.0)

            heat = to_heatmap(p0_01, H, W)
            over = overlay(base, heat, OVERLAY_ALPHA)

            cv2.imwrite(str(out_dir / "gdmp_pred_heat.png"), heat)
            cv2.imwrite(str(out_dir / "gdmp_pred_overlay.png"), over)
        else:
            print(f"[WARN] GDMP last_pgdp_p0 not found for {cond}")

        # ---- GDMP GT (RIGHT, gray) ----
        # trainer.py에서처럼 GT builder 사용: {"M_gt":(B,C,H,W), "weights":..., "valid":...}
        gt = pgdp_gt_builder(anno) if callable(pgdp_gt_builder) else pgdp_gt_builder.build(anno)
        M_gt = gt["M_gt"][0].detach().float().cpu().numpy()  # (C,H,W)
        gt_max = M_gt.max(axis=0)  # (H,W)
        gt01 = np.clip(gt_max, 0, 1)  # 원래 0~1 prior
        gt_gray = to_gray_png(gt01, H, W)
        cv2.imwrite(str(out_dir / "pgdp_gt_gray.png"), gt_gray)

        tmp_list.unlink(missing_ok=True)

    print(f"[DONE] SDUP/GDMP maps saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()