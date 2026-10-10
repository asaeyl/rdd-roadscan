"""Builds a tiny fake dataset in the same layout as the Kaggle RDD2022 copy (random split over consecutive frames)."""
import random
from pathlib import Path

import numpy as np
from PIL import Image

COUNTRIES = [("Japan", 900, .15), ("India", 700, .15), ("Czech", 300, .02), ("Norway", 800, .10),
             ("United_States", 500, .10), ("China_Drone", 250, .20), ("China_MotorBike", 200, .20)]


def build_mock_dataset(root, scale=1.0, seed=1):
    """root/{train,val,test}/{images,labels}/COUNTRY_000123.{jpg,txt}. Returns root."""
    root = Path(root)
    for split in ["train", "val", "test"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    np.random.seed(seed)
    for country, n, p_d20 in COUNTRIES:
        for i in range(int(n * scale)):
            split = rng.choices(["train", "val", "test"], [.7, .15, .15])[0]
            name = f"{country}_{i:06d}"
            lines = []
            if rng.random() > .15:
                for class_id, p in zip(range(4), [.5, .35, p_d20, .15]):
                    if rng.random() < p:
                        lines += [f"{class_id} 0.5 0.5 0.1 0.2"] * rng.randint(1, 3)
            Image.fromarray((np.random.rand(16, 16, 3) * 255).astype("uint8")).save(root / split / "images" / f"{name}.jpg")
            (root / split / "labels" / f"{name}.txt").write_text("\n".join(lines))
    return root
