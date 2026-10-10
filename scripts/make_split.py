"""Create the frozen 8,000-image split ONCE.

    python scripts/make_split.py

The result (data/splits_8000.csv) is committed to GitHub. E1, E2, E3 ... only READ it,
so every experiment uses exactly the same images. The script refuses to overwrite an existing split
(use --force only if you really want a new sample, and then re-run ALL experiments).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import random

from src import config as cfg
from src import data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    parser.add_argument("--target-images", type=int, default=cfg.TARGET_IMAGES)
    parser.add_argument("--frames-per-block", type=int, default=cfg.FRAMES_PER_BLOCK)
    args = parser.parse_args()

    if cfg.SPLIT_CSV.exists() and not args.force:
        sys.exit(f"{cfg.SPLIT_CSV} already exists. All experiments share it - not overwriting. (use --force to replace)")

    random.seed(cfg.SEED); np.random.seed(cfg.SEED)
    dataset_dir = data.get_dataset_dir()
    print("Dataset folder:", dataset_dir)

    all_images = data.load_all_images(dataset_dir)
    print(len(all_images), "images found")

    subset = data.make_balanced_subset(all_images, args.target_images)
    subset["split"] = data.assign_blocks_to_splits(subset, frames_per_block=args.frames_per_block)

    print("\nImages per country and damage type:")
    print(subset.groupby("country")[cfg.HAS_COLUMNS + ["has_no_damage"]].sum().assign(total=subset.groupby("country").size()))
    print("\nShare per split:\n", subset.split.value_counts(normalize=True).round(3))
    print("\nImages per split and damage type:\n", subset.groupby("split")[cfg.HAS_COLUMNS + ["has_no_damage"]].sum())
    print(f"\nTest images with a near-copy in train: {data.share_of_test_with_near_copy_in_train(subset):.1%}"
          "   (if clearly above ~3%, increase --frames-per-block)")

    path = data.save_split_csv(subset)
    print("\nSaved", path, "| split id:", data.split_fingerprint(subset))


if __name__ == "__main__":
    main()
