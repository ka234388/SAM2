# EVALUATE.PY
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# === Logger
def create_logger(log_dir: Path, name="evaluate", level="INFO"):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    ch = logging.StreamHandler()
    fh = logging.FileHandler(log_dir / f"{name}.log", mode="w")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    ch.setFormatter(fmt); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger

# === Overlay function
def overlay_mask(img, mask, color=(0,255,0), alpha=0.4):
    if mask.ndim == 3:
        mask = np.array(Image.fromarray(mask).convert("L"))
    mask_bool = mask > 127
    out = img.copy()
    col = np.array(color, dtype=np.uint8)
    out[mask_bool] = (col * alpha + out[mask_bool] * (1 - alpha)).astype(np.uint8)
    return out

# === Display function
def show_result(row, paths, use_overlays=True):
    img_name = row["image"]
    stem = Path(img_name).stem

    img_path = paths["img_dir"] / img_name
    lbl_path = paths["lbl_dir"] / f"{stem}.png"
    if not lbl_path.exists():
        lbl_path = paths["lbl_dir"] / f"{stem}_L.png"

    # Masks
    b_people = np.array(Image.open(paths["mask_b"] / f"{stem}_people.png"))
    b_vehicle= np.array(Image.open(paths["mask_b"] / f"{stem}_vehicle.png"))
    i_people = np.array(Image.open(paths["mask_i"] / f"{stem}_people.png"))
    i_vehicle= np.array(Image.open(paths["mask_i"] / f"{stem}_vehicle.png"))

    img = np.array(Image.open(img_path).convert("RGB"))
    gt  = np.array(Image.open(lbl_path).convert("L"))

    if use_overlays:
        P_BASE = overlay_mask(img, b_people, color=(0,255,0))
        P_IMPR = overlay_mask(img, i_people, color=(0,200,0))
        V_BASE = overlay_mask(img, b_vehicle, color=(0,0,255))
        V_IMPR = overlay_mask(img, i_vehicle, color=(0,0,200))
        cmap=None
    else:
        P_BASE, P_IMPR, V_BASE, V_IMPR = b_people, i_people, b_vehicle, i_vehicle
        cmap="gray"

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax = ax.ravel()
    ax[0].imshow(img); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(gt, cmap="gray"); ax[1].set_title("GT label"); ax[1].axis("off")
    ax[2].imshow(P_BASE, cmap=cmap); ax[2].set_title(f"People (Baseline)\nDice={row['dice_people_baseline']:.3f}"); ax[2].axis("off")
    ax[3].imshow(P_IMPR, cmap=cmap); ax[3].set_title(f"People (Improved)\nDice={row['dice_people_improved']:.3f}"); ax[3].axis("off")
    ax[4].imshow(V_BASE, cmap=cmap); ax[4].set_title(f"Vehicle (Baseline)\nDice={row['dice_vehicle_baseline']:.3f}"); ax[4].axis("off")
    ax[5].imshow(V_IMPR, cmap=cmap); ax[5].set_title(f"Vehicle (Improved)\nDice={row['dice_vehicle_improved']:.3f}"); ax[5].axis("off")

    plt.tight_layout()
    plt.show()

# === Paths
OUT_ROOT = Path("/kaggle/working/a3_sam2_camvid")
IMG_DIR = Path("/kaggle/input/camvid/CamVid/val")
LBL_DIR = Path("/kaggle/input/camvid/CamVid/val_labels")
CSV_PATH = OUT_ROOT / "camvid_val_dice.csv"
MASK_B = OUT_ROOT / "masks_baseline"
MASK_I = OUT_ROOT / "masks_improved"
LOG_DIR = OUT_ROOT / "logs"

logger = create_logger(LOG_DIR)
logger.info("Starting selected image display")

# Load CSV
df = pd.read_csv(CSV_PATH)

# Specify images to display
selected_images = ["0001TP_009900.png", "0001TP_009060.png"]
subset = df[df["image"].isin(selected_images)]
logger.info(f"Displaying selected images: {selected_images}")

paths = {
    "img_dir": IMG_DIR,
    "lbl_dir": LBL_DIR,
    "mask_b": MASK_B,
    "mask_i": MASK_I,
}

# Show each selected image
for _, row in subset.iterrows():
    logger.info(f"Displaying {row['image']}")
    try:
        show_result(row, paths, use_overlays=True)
    except Exception as e:
        logger.error(f"Failed to display {row['image']}: {e}")
