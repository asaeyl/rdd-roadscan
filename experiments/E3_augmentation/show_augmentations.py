"""E3 - Augmentation viewer.

Shows what each augmentation variant in run.py actually looks like on a real
road-damage image, so the 'our hypothesis' column has a picture to go with it.

    python experiments/E3_augmentation/show_augmentations.py
    python experiments/E3_augmentation/show_augmentations.py --images-dir /content/rdd-roadscan/work/yolo/images
    python experiments/E3_augmentation/show_augmentations.py --seed 1

Where the photos come from (first one that works):
  1. --images-dir            any folder that contains the .jpg files (searched recursively)
  2. work/yolo/images        the YOLO folder that E1/E2/E3 build (exists after any experiment ran)
  3. the RDD2022 dataset     RDD_DATA_DIR if set, otherwise the Kaggle download (the same one make_split.py uses)
The split CSV only holds image NAMES (e.g. Japan_000734), not paths - the names are matched to the files found above.
Only images that contain damage are used; a pothole image (D40) is preferred as the main example.

For each variant in VARIANTS it writes one PNG into
    experiments/E3_augmentation/aug_preview/<variant>.png

The pictures are produced with OpenCV and mirror the *visual meaning* of the
Ultralytics settings (mosaic, hsv_*, fliplr, flipud, degrees). They are an
illustration, not the exact RNG of Ultralytics training (scale / translate are not shown).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src import data

DEFAULT_CSV = Path(cfg.SPLIT_CSV)
IMG_SIZE = int(cfg.IMAGE_SIZE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
GREY = (114, 114, 114)                      # the padding colour Ultralytics uses

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "aug_preview"

# Must stay in sync with VARIANTS in run.py
VARIANTS = {
    "default":   "reference: YOLO defaults (mosaic + hsv + fliplr)",
    "no_mosaic": "mosaic OFF -> single image",
    "no_hsv":    "colour jitter OFF (hue / saturation / value)",
    "no_fliplr": "horizontal flip OFF",
    "flipud_on": "vertical flip ON (UNREALISTIC)",
    "rotate_30": "rotation up to 30 deg (UNREALISTIC for dashcam)",
}


# ---------------------------------------------------------------- args

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--images-dir", default=None,
                    help="folder with the photos (searched recursively); default: work/yolo/images, then the RDD2022 dataset")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ---------------------------------------------------------------- finding the photos

def index_folder(folder):
    """{file name without extension: path} for every photo below `folder`."""
    return {p.stem: p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}


def build_lookup(images_dir):
    """Find a folder with photos. Returns (index, description of where they were found)."""
    candidates = []
    if images_dir:
        candidates.append((f"--images-dir {images_dir}", Path(images_dir)))
    candidates.append(("YOLO folder built by the experiments", cfg.WORK_DIR / "yolo" / "images"))

    for label, folder in candidates:
        if folder.is_dir():
            index = index_folder(folder)
            if index:
                return index, f"{label} ({len(index)} photos)"
            print(f"  (no photos inside {folder})")
        else:
            print(f"  (folder not found: {folder})")

    print("  -> using the RDD2022 dataset (RDD_DATA_DIR, or the Kaggle download: this can take a few minutes the first time)")
    dataset_dir = data.get_dataset_dir()
    index = {p.stem: p for p in Path(dataset_dir).glob("*/images/*") if p.suffix.lower() in IMAGE_SUFFIXES}
    return index, f"RDD2022 dataset {dataset_dir} ({len(index)} photos)"


def find_images(csv_path, images_dir, n=4, seed=0):
    df = pd.read_csv(csv_path)
    name_col = next((c for c in ("image_path", "path", "name", "filename") if c in df.columns), None)
    if name_col is None:
        raise SystemExit(f"No image name column in {csv_path} (columns: {list(df.columns)})")

    df = df.assign(stem=df[name_col].astype(str).map(lambda v: Path(v).stem))
    if "n_boxes" in df.columns:
        df = df[df["n_boxes"] > 0]                                        # only photos that show damage
    df = df.sample(frac=1.0, random_state=seed)
    if "has_D40" in df.columns:                                           # pothole photos first: easiest to judge
        df = pd.concat([df[df["has_D40"] == 1], df[df["has_D40"] != 1]])

    index, where = build_lookup(images_dir)
    print("Photos found in:", where)
    found = [index[s] for s in df["stem"] if s in index][:n]
    if not found:
        examples = list(df["stem"].head(3))
        raise SystemExit(
            f"None of the {len(df)} photos named in {csv_path} (e.g. {examples}) exist in: {where}.\n"
            "Pass the folder that really contains them, e.g. --images-dir /content/rdd-roadscan/work/yolo/images "
            "or the RDD_SPLIT folder of the Kaggle download.")
    return found


def load_image(path, size=IMG_SIZE):
    """Read a photo and fit it into a size x size square WITHOUT stretching (grey padding, like YOLO's letterbox)."""
    im = cv2.imread(str(path))
    if im is None:
        return None
    h, w = im.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(im, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), GREY, dtype=np.uint8)
    top, left = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


# ---------------------------------------------------------------- augmentations

def hsv_jitter(im, rng, h=0.015, s=0.7, v=0.4):
    """Mirror of Ultralytics' RandomHSV (the hsv_h / hsv_s / hsv_v settings)."""
    gains = rng.uniform(-1, 1, 3) * np.array([h, s, v]) + 1
    hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=np.float32)
    lut_hue = ((x * gains[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * gains[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * gains[2], 0, 255).astype(np.uint8)
    out = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)


def rotate(im, deg):
    """Rotate around the centre; the corners are filled with grey like in YOLO training."""
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=GREY)


def mosaic4(images, size=IMG_SIZE):
    """Simple quadrant mosaic, visual approximation of YOLO's Mosaic."""
    if len(images) < 4:
        images = (list(images) * 4)[:4]
    half = size // 2
    q = [cv2.resize(im, (half, half), interpolation=cv2.INTER_AREA) for im in images[:4]]
    return np.vstack([np.hstack([q[0], q[1]]), np.hstack([q[2], q[3]])])


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- per-variant panels

def panels_for(variant, imgs, rng):
    """Return [(bgr_image, subtitle), ...] to show for this variant."""
    base = imgs[0]

    if variant == "default":
        m = hsv_jitter(mosaic4(imgs), rng)
        if rng.random() < 0.5:
            m = cv2.flip(m, 1)
        return [(m, "default: mosaic + HSV + fliplr")]

    if variant == "no_mosaic":
        m = hsv_jitter(mosaic4(imgs), rng)
        s = hsv_jitter(base, rng)
        return [(m, "default: mosaic ON"), (s, "no_mosaic: single image")]

    if variant == "no_hsv":
        m = hsv_jitter(base, rng)
        return [(m, "default: HSV jitter ON"), (base, "no_hsv: no colour jitter")]

    if variant == "no_fliplr":
        return [(cv2.flip(base, 1), "default: fliplr ON"), (base, "no_fliplr: original")]

    if variant == "flipud_on":
        return [(base, "default: no v-flip"),
                (cv2.flip(base, 0), "flipud_on: vertical flip")]

    if variant == "rotate_30":
        deg = float(rng.uniform(-30, 30))
        return [(base, "default: 0 deg"),
                (rotate(base, deg), f"rotate_30: {deg:+.1f} deg")]

    raise ValueError(f"unknown variant: {variant}")


def save_variant(variant, imgs, out_dir, seed):
    rng = np.random.default_rng(seed)
    panels = panels_for(variant, imgs, rng)

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (img, subtitle) in zip(axes, panels):
        ax.imshow(_rgb(img))
        ax.set_title(subtitle, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"E3 - {variant}: {VARIANTS[variant]}", fontsize=12)
    plt.tight_layout()

    out_path = out_dir / f"{variant}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = find_images(args.csv, args.images_dir, n=4, seed=args.seed)
    print("Sample images (the first one is used for the single-image panels):")
    for p in paths:
        print("  ", p)

    imgs = [im for im in (load_image(p) for p in paths) if im is not None]
    if not imgs:
        raise SystemExit("Could not read any of the sample images.")

    for variant in VARIANTS:
        out = save_variant(variant, imgs, out_dir, seed=args.seed)
        print("saved:", out)

    print(f"\nAll previews written to {out_dir}/")


if __name__ == "__main__":
    main()
