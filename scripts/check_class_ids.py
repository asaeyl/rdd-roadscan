"""Draw example images so you can SEE what the class ids mean (run once, look at the picture).

    python scripts/check_class_ids.py        ->  data/class_id_check.png

Top row: images with class id 3 (should be potholes = D40).  Bottom row: class id 0 (thin longitudinal cracks = D00).
If it looks different, fix CLASS_ID_TO_NAME in src/config.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src import config as cfg
from src import data


def main():
    dataset_dir = data.get_dataset_dir()
    rows = [{"name": p.stem, "original_split": s} for s in ["train", "val", "test"]
            for p in sorted((dataset_dir / s / "labels").glob("*.txt"))]
    table = data.attach_paths(pd.DataFrame(rows), dataset_dir)
    print("Class ids found:\n", pd.Series([b[0] for boxes in table.boxes for b in boxes]).value_counts().sort_index())

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for axes_row, class_id in zip(axes, [3, 0]):
        has_it = table.boxes.map(lambda boxes: any(b[0] == class_id for b in boxes))
        for ax, (_, row) in zip(axes_row, table[has_it].sample(4, random_state=cfg.SEED).iterrows()):
            image = Image.open(row.image_path)
            width, height = image.size
            ax.imshow(image); ax.axis("off"); ax.set_title(f"{row['name']} | id {class_id}")
            for (cid, x, y, w, h) in row.boxes:
                ax.add_patch(patches.Rectangle(((x - w / 2) * width, (y - h / 2) * height), w * width, h * height, fill=False,
                                               linewidth=2, edgecolor="red" if cid == class_id else "yellow"))
    out = cfg.DATA_DIR / "class_id_check.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
