import json
import random
from pathlib import Path
from collections import defaultdict

import torch

from tools.dataset import Dataset
from clft.clft import CLFT


# ==========================================================
# USER CONFIG
# ==========================================================
CONFIG_PATH = "/home/john/dev_ws/CLFT/config.json"
CKPT_PATH = "/home/john/dev_ws/CLFT/logs/clft_ctca_v4_fusioncheckpoint_89_9210_6911.pth"

LABELED_ROOT = Path("/home/john/dev_ws/CLFT/waymo_dataset/labeled")  # day/night 아래가 있음
SPLIT = "val"          # "train" or "val"
K_PER_COND = 20        # 조건별 랜덤 샘플 개수
SEED = 0
PRINT_EACH = False     # True면 샘플별 gate 로그도 다 출력
TMP_LIST_PATH = "/tmp/_gate_demo_list.txt"
# ==========================================================


CONDITIONS = [
    ("day", "rain"),
    ("day", "not_rain"),
    ("night", "rain"),
    ("night", "not_rain"),
]


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
    state = ckpt["model_state_dict"]
    model.load_state_dict(state, strict=True)


def discover_camera_pngs(daynight: str, rainflag: str):
    """
    labeled/{day|night}/{rain|not_rain}/camera/**/*.png 를 모두 찾는다.
    Dataset이 내부에서 ./waymo_dataset/ + line 을 붙여 읽는 구조이므로,
    line은 'labeled/...' 상대경로 형태로 만든다.
    """
    cam_dir = LABELED_ROOT / daynight / rainflag / "camera"
    if not cam_dir.exists():
        raise FileNotFoundError(f"camera dir not found: {cam_dir}")

    files = sorted(cam_dir.rglob("*.png"))
    rels = [f"labeled/{daynight}/{rainflag}/camera/{p.relative_to(cam_dir).as_posix()}" for p in files]
    return rels


def make_temp_list(lines):
    p = Path(TMP_LIST_PATH)
    p.write_text("\n".join(lines) + "\n")
    return str(p)


@torch.no_grad()
def run_one(model, ds, idx: int, device: str):
    sample = ds[idx]
    rgb = sample["rgb"].unsqueeze(0).to(device)
    lidar = sample["lidar"].unsqueeze(0).to(device)

    _out_depth, _out_seg = model(rgb, lidar, modal="cross_fusion")

    g_list, s_list = [], []
    stage_logs = []
    for si, fus in enumerate(model.fusions):
        d = getattr(fus, "last_log", None) or {}
        # last_log 키 자동 대응
        if "g" in d and "scale" in d:
            g = float(d["g"]); s = float(d["scale"])
        elif "g_mean" in d and "s_mean" in d:
            g = float(d["g_mean"]); s = float(d["s_mean"])
        else:
            # eval에서 last_log 갱신이 안 되는 경우
            g = float("nan"); s = float("nan")

        g_list.append(g)
        s_list.append(s)
        stage_logs.append(f"S{si}: g={g:.3f} scale={s:.3f}")

    return g_list, s_list, " | ".join(stage_logs)


def mean_std(xs):
    xs = [x for x in xs if x == x]  # NaN 제거
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(1, (len(xs) - 1))
    return m, v ** 0.5


def main():
    random.seed(SEED)

    cfg = json.loads(Path(CONFIG_PATH).read_text())
    device = cfg["General"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu"

    # 1) 조건별 파일 수집 + K개 샘플링
    picked = []  # (cond_name, rel_path)
    for dn, rf in CONDITIONS:
        rels = discover_camera_pngs(dn, rf)
        if len(rels) == 0:
            raise RuntimeError(f"No pngs found for {dn}/{rf}")
        k = min(K_PER_COND, len(rels))
        sampled = random.sample(rels, k)
        cond_name = f"{dn}/{rf}"
        picked.extend([(cond_name, r) for r in sampled])

    # 2) Dataset은 list 파일을 받으므로 임시 list 생성
    lines = [r for _cond, r in picked]
    tmp_list = make_temp_list(lines)

    ds = Dataset(config=cfg, split=SPLIT, path=tmp_list)

    # 3) 모델 로드
    model = build_model(cfg, device)
    load_ckpt(model, CKPT_PATH, device)
    model.eval()

    print(f"[INFO] device={device} split={SPLIT} seed={SEED} K_PER_COND={K_PER_COND} total_samples={len(ds)}")
    print(f"[INFO] labeled_root={LABELED_ROOT}")
    print(f"[INFO] tmp_list={tmp_list}")

    # 4) inference + 통계 집계
    stats_g = defaultdict(lambda: defaultdict(list))
    stats_s = defaultdict(lambda: defaultdict(list))

    for i, (cond_name, rel) in enumerate(picked):
        g_list, s_list, stage_log = run_one(model, ds, i, device)

        for si, (g, s) in enumerate(zip(g_list, s_list)):
            stats_g[cond_name][si].append(g)
            stats_s[cond_name][si].append(s)

        if PRINT_EACH:
            print(f"[{i}] {rel}\n  {stage_log}\n")

    # 5) 조건별 평균/표준편차 출력
    print("\n=== [GATE STATS] mean±std over samples (per condition) ===")
    for dn, rf in CONDITIONS:
        cond_name = f"{dn}/{rf}"
        parts = []
        for si in range(len(model.fusions)):
            gm, gs = mean_std(stats_g[cond_name][si])
            sm, ss = mean_std(stats_s[cond_name][si])
            parts.append(f"S{si}: g={gm:.3f}±{gs:.3f} scale={sm:.3f}±{ss:.3f}")
        print(f"[{cond_name}] " + " | ".join(parts))


if __name__ == "__main__":
    main()
