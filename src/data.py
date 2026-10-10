"""Everything about the data.

1. read RDD2022 into one table (one row per image)
2. build a balanced subset (by country and damage type)
3. split it without leakage (whole blocks of consecutive frames stay together)
4. save the split ONCE (save_split_csv) and let every experiment READ it (load_split_csv)
5. build the YOLO folder (images + label files) from the saved split
"""
import hashlib
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src import config as cfg


# ============================================================================ 1. read the dataset
def get_dataset_dir():
    """Folder that contains train/ val/ test/ (each with images/ and labels/).
    Set the environment variable RDD_DATA_DIR to use a local copy, otherwise it is downloaded from Kaggle."""
    import os
    if os.environ.get("RDD_DATA_DIR"):
        return Path(os.environ["RDD_DATA_DIR"])
    import kagglehub
    return Path(kagglehub.dataset_download("aliabdelmenam/rdd-2022")) / "RDD_SPLIT"


def read_label_file(label_path):
    """Read one YOLO label file. Returns a list of boxes: (class_id, x_center, y_center, width, height)."""
    boxes = []
    for line in Path(label_path).read_text().splitlines():
        if line.strip():
            class_id, x, y, w, h = line.split()[:5]
            boxes.append((int(class_id), float(x), float(y), float(w), float(h)))
    return boxes


def attach_paths(table, dataset_dir, read_boxes=True):
    """Add image_path and label_path (and the boxes) to a table that has the columns name + original_split."""
    table = table.copy()
    dataset_dir = Path(dataset_dir)
    table["image_path"] = [str(dataset_dir / s / "images" / f"{n}.jpg") for s, n in zip(table.original_split, table.name)]
    table["label_path"] = [str(dataset_dir / s / "labels" / f"{n}.txt") for s, n in zip(table.original_split, table.name)]
    if read_boxes:
        table["boxes"] = [read_label_file(p) for p in table.label_path]
    return table


def add_damage_columns(table):
    """Add: count_D00.., has_D00.., n_boxes, has_no_damage."""
    table["damage_names"] = table["boxes"].map(
        lambda boxes: [cfg.CLASS_ID_TO_NAME[b[0]] for b in boxes if b[0] in cfg.CLASS_ID_TO_NAME])
    for name in cfg.DAMAGE_CLASSES:
        table["count_" + name] = table["damage_names"].map(lambda names: names.count(name))
        table["has_" + name] = (table["count_" + name] > 0).astype(int)
    table["n_boxes"] = table[["count_" + n for n in cfg.DAMAGE_CLASSES]].sum(axis=1)
    table["has_no_damage"] = (table["boxes"].map(len) == 0).astype(int)     # label file is completely empty
    return table


def load_all_images(dataset_dir):
    """Go through the Kaggle train/val/test folders and return one row per image."""
    rows = []
    for original_split in ["train", "val", "test"]:
        for label_path in sorted((Path(dataset_dir) / original_split / "labels").glob("*.txt")):
            if not (Path(dataset_dir) / original_split / "images" / (label_path.stem + ".jpg")).exists():
                continue
            country, frame_number = label_path.stem.rsplit("_", 1)         # "China_Drone_000123" -> "China_Drone", "000123"
            rows.append({"country": country, "frame_number": int(frame_number),
                         "name": label_path.stem, "original_split": original_split})
    return add_damage_columns(attach_paths(pd.DataFrame(rows), dataset_dir))


# ============================================================================ 2. balanced subset
def split_quota_between_countries(images_available, total_wanted):
    """Give every country the same quota; countries that run out give their unused share to the others.
    images_available: {"Japan": 10000, ...}  ->  returns {"Japan": 1334, ...}"""
    quota = {country: 0 for country in images_available}
    still_open = set(images_available)
    remaining = total_wanted
    while remaining > 0 and still_open:
        equal_share = max(1, remaining // len(still_open))
        for country in list(still_open):
            give = min(equal_share, images_available[country] - quota[country], remaining)
            quota[country] += give
            remaining -= give
            if quota[country] >= images_available[country]:
                still_open.discard(country)
    return quota


def pick_images_for_country(country_images, how_many, rng, share_without_damage=cfg.SHARE_WITHOUT_DAMAGE):
    """Pick `how_many` images from one country, keeping the 4 damage types as equal as possible."""
    # 1) images without damage
    n_without_damage = int(round(how_many * share_without_damage))
    no_damage_pool = country_images[country_images.has_no_damage == 1]
    no_damage_picked = no_damage_pool.sample(min(n_without_damage, len(no_damage_pool)), random_state=cfg.SEED)

    # 2) images with damage: always serve the damage type that has the fewest images so far
    pool = country_images[country_images.n_boxes > 0].copy()
    images_per_class = {name: 0 for name in cfg.DAMAGE_CLASSES}
    picked_index = []
    while len(picked_index) < how_many - len(no_damage_picked) and len(pool) > 0:
        classes_still_available = [n for n in cfg.DAMAGE_CLASSES if pool["has_" + n].sum() > 0]
        if not classes_still_available:
            break
        rarest = min(classes_still_available, key=lambda n: images_per_class[n])
        candidates = pool[pool["has_" + rarest] == 1]
        # prefer images with few boxes, so crowded crack images do not dominate
        chosen = candidates.sample(1, weights=1 / candidates.n_boxes.clip(lower=1),
                                   random_state=int(rng.integers(10**9))).iloc[0]
        picked_index.append(chosen.name)
        pool = pool.drop(chosen.name)
        for n in cfg.DAMAGE_CLASSES:
            images_per_class[n] += int(chosen["has_" + n] > 0)
    return pd.concat([no_damage_picked, country_images.loc[picked_index]])


def make_balanced_subset(all_images, target_images=cfg.TARGET_IMAGES, seed=cfg.SEED):
    """Balanced by country (equal quota) and by damage type (rarest class first)."""
    rng = np.random.default_rng(seed)
    quota = split_quota_between_countries(all_images.country.value_counts().to_dict(), target_images)
    parts = [pick_images_for_country(all_images[all_images.country == country], n, rng) for country, n in quota.items()]
    return pd.concat(parts).reset_index(drop=True)


# ============================================================================ 3. leak-free split
def assign_blocks_to_splits(table, frames_per_block=cfg.FRAMES_PER_BLOCK, split_shares=cfg.SPLIT_SHARES, seed=cfg.SEED):
    """Return a Series with 'train' / 'val' / 'test' for every image. Whole blocks stay together."""
    block_of_image = table.country + "_" + (table.frame_number // frames_per_block).astype(str)
    split_of_block = {}
    for country, country_images in table.groupby("country"):
        block_sizes = country_images.groupby(block_of_image[country_images.index]).size().to_dict()
        blocks = sorted(block_sizes)                                          # fixed order, then shuffle with the seed
        random.Random(seed).shuffle(blocks)
        wanted = {s: share * len(country_images) for s, share in split_shares.items()}
        filled = {s: 0 for s in split_shares}
        for block in sorted(blocks, key=lambda b: -block_sizes[b]):           # biggest blocks first (stable sort keeps the shuffle for ties)
            split = max(split_shares, key=lambda s: wanted[s] - filled[s])    # split furthest below its target
            split_of_block[block] = split
            filled[split] += block_sizes[block]
    return block_of_image.map(split_of_block)


def image_fingerprint(path):
    """Tiny 64-bit fingerprint of an image (dHash). Near-identical photos have almost the same fingerprint."""
    small = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten()


def share_of_test_with_near_copy_in_train(table):
    """Leak check: which share of the test images has a near-copy in train? (aim: below about 3%)"""
    train_prints = np.stack([image_fingerprint(p) for p in table[table.split == "train"].image_path])
    test_prints = [image_fingerprint(p) for p in table[table.split == "test"].image_path]
    differences = [(train_prints != fp).sum(axis=1).min() for fp in test_prints]    # fewer than 5 differing bits = near copy
    return float(np.mean(np.array(differences) <= 4))


# ============================================================================ 4. save ONCE, load in EVERY experiment
SPLIT_COLUMNS = (["country", "frame_number", "name", "original_split", "split", "n_boxes", "has_no_damage"]
                 + ["count_" + n for n in cfg.DAMAGE_CLASSES] + cfg.HAS_COLUMNS)


def save_split_csv(subset, path=cfg.SPLIT_CSV):
    """Write the frozen split. The file only holds names + labels + split (no absolute paths), so it works on any machine."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    subset[SPLIT_COLUMNS].to_csv(path, index=False)
    return Path(path)


def load_split_csv(path=cfg.SPLIT_CSV):
    """Read the frozen split (used by E1, E2, E3, ... so they all use exactly the same images)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Create it once with:  python scripts/make_split.py")
    return pd.read_csv(path)


def split_fingerprint(split_table):
    """Short ID of the split (hash of image names + their split). Printed in every results file,
    so you can prove that two experiments used the identical sample."""
    lines = sorted(f"{n}:{s}" for n, s in zip(split_table["name"], split_table["split"]))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]


# ============================================================================ 5. YOLO folder from the saved split
def build_yolo_dataset(split_table, out_dir=None, force=False):
    """Copy the images and write YOLO label files (class ids renumbered 0-3) for the split in `split_table`.
    split_table must have image_path, label_path (see attach_paths). Returns the path of the YAML file for YOLO."""
    out_dir = Path(out_dir or cfg.WORK_DIR / "yolo")
    yaml_path = out_dir / "rdd8k.yaml"
    n_expected = len(split_table)
    n_found = sum(1 for _ in (out_dir / "images").rglob("*.jpg")) if (out_dir / "images").exists() else 0
    if yaml_path.exists() and n_found == n_expected and not force:
        return yaml_path                                                     # already built

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in cfg.SPLIT_SHARES:
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

    for row in split_table.itertuples():
        shutil.copy(row.image_path, out_dir / "images" / row.split / f"{row.name}.jpg")
        lines = []
        for (class_id, x, y, w, h) in (row.boxes if hasattr(row, "boxes") else read_label_file(row.label_path)):
            if class_id in cfg.CLASS_ID_TO_NAME:
                new_id = cfg.DAMAGE_CLASSES.index(cfg.CLASS_ID_TO_NAME[class_id])
                lines.append(f"{new_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        (out_dir / "labels" / row.split / f"{row.name}.txt").write_text("\n".join(lines))

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(cfg.DAMAGE_CLASSES))
    yaml_path.write_text(f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    return yaml_path
