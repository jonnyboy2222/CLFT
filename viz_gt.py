import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# 1) 경로 설정
# -----------------------------
BASE = Path("/home/john/dev_ws/CLFT/waymo_dataset/labeled")

samples = [
    ("day/not_rain",
     "segment-30779396576054160_1880_000_1900_000_with_camera_labels_0000000041.png"),
    ("day/rain",
     "segment-89454214745557131_3160_000_3180_000_with_camera_labels_0000000000.png"),
    ("night/not_rain",
     "segment-10226164909075980558_180_000_200_000_with_camera_labels_0000000187.png"),
    ("night/rain",
     "segment-17791493328130181905_1480_000_1500_000_with_camera_labels_0000000185.png"),
]

# annotation 폴더 아래
paths = [BASE / split / "annotation" / fname for split, fname in samples]

# -----------------------------
# 2) 라벨 팔레트
# -----------------------------
CLASS_NAMES = {
    0: "class0",
    1: "class1",
    2: "class2",
    3: "class3",
    4: "class4",
    5: "class5",
}

# (R, G, B) 팔레트 (가시성 좋은 임의값)
PALETTE = {
    0: (80, 80, 80),        # background / sky / empty -> BLACK
    1: (0, 255, 0),    # vehicle                  -> GREEN
    2: (255, 0, 0),    # human (ped + cyclist)    -> RED
    3: (0, 0, 0),     # ignore (road / pole ...) -> DARK GRAY
    4: (255, 0, 0),    # human (ped + cyclist)    -> RED
}


def colorize_label(label_u8: np.ndarray, palette: dict[int, tuple[int,int,int]]) -> np.ndarray:
    """label_u8: (H,W) uint8 -> color: (H,W,3) uint8"""
    h, w = label_u8.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, rgb in palette.items():
        out[label_u8 == k] = rgb
    return out

def load_label_png(p: Path) -> np.ndarray:
    im = Image.open(p)
    # 라벨 png는 보통 single-channel (L) 이거나 palette(P)인데
    # 어떤 모드든 최종적으로 (H,W) uint8 라벨로 변환
    lab = np.array(im)
    if lab.ndim == 3:
        # 혹시 RGB로 저장된 특이 케이스면 첫 채널만 사용(필요시 수정)
        lab = lab[:, :, 0]
    return lab.astype(np.uint8)

# -----------------------------
# 3) 4장 한 번에 보기 + 통계 출력
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
axes = axes.flatten()

for ax, (split, fname), p in zip(axes, samples, paths):
    if not p.exists():
        ax.set_title(f"[MISSING] {split}\n{fname}", fontsize=10)
        ax.axis("off")
        continue

    lab = load_label_png(p)
    lab = lab[160:320, 0:480]
    color = colorize_label(lab, PALETTE)

    uniq = np.unique(lab)
    uniq_str = ", ".join(str(int(x)) for x in uniq)

    ax.imshow(color)
    ax.set_title(f"{split}\n{fname}\nunique labels: {uniq_str}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# 4) 컬러 GT 저장(원하면 사용)
# -----------------------------
SAVE_DIR = BASE / "_vis_gt"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

for (split, fname), p in zip(samples, paths):
    if not p.exists():
        continue
    lab = load_label_png(p)
    lab = lab[160:320, 0:480]
    color = colorize_label(lab, PALETTE)
    out_path = SAVE_DIR / f"{split.replace('/', '_')}__{fname.replace('.png','')}_color.png"
    Image.fromarray(color).save(out_path)
    print("saved:", out_path)
