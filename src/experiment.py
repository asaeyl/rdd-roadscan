"""Common start-up for every experiment script: seed, arguments, the SAME split, the YOLO folder, output folders."""
import argparse
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg
from src import data


@dataclass
class Setup:
    name: str
    args: argparse.Namespace
    epochs: int
    smoke: bool
    results_dir: Path
    weights_dir: Path
    split_table: pd.DataFrame          # the frozen split, with image paths and boxes
    data_yaml: Path                    # YOLO dataset description
    split_id: str                      # short ID of the split (identical in every experiment)
    yolo_extra: dict = field(default_factory=dict)

    def part(self, split):
        """Rows of one split: 'train', 'val' or 'test'."""
        return self.split_table[self.split_table.split == split].reset_index(drop=True)


def set_seed(seed=cfg.SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def start_experiment(name, argv=None, extra_args=None):
    """Read arguments, fix the seed, load the frozen split, build the YOLO folder, create output folders."""
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS, help="number of epochs (default from config)")
    parser.add_argument("--smoke", action="store_true", help="quick test: 1 epoch, 5%% of the YOLO data, outputs go to *_smoke folders")
    parser.add_argument("--retrain", action="store_true", help="train again even if saved weights exist")
    for flag, kwargs in (extra_args or {}).items():
        parser.add_argument(flag, **kwargs)
    args = parser.parse_args(argv)

    set_seed()
    smoke = args.smoke
    epochs = 1 if smoke else args.epochs
    suffix = "_smoke" if smoke else ""
    exp_dir = cfg.EXPERIMENTS_DIR / name
    results_dir, weights_dir = exp_dir / f"results{suffix}", exp_dir / f"weights{suffix}"
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    split_table = data.load_split_csv()                                     # <- THE shared sample
    split_id = data.split_fingerprint(split_table)
    split_table = data.attach_paths(split_table, data.get_dataset_dir())
    data_yaml = data.build_yolo_dataset(split_table)

    print(f"== {name} | split id {split_id} | {len(split_table)} images | epochs {epochs}{' (SMOKE TEST)' if smoke else ''}")
    return Setup(name, args, epochs, smoke, results_dir, weights_dir, split_table, data_yaml, split_id,
                 yolo_extra={"fraction": 0.05} if smoke else {})


def common_settings(setup, **more):
    """Settings written at the top of every results file."""
    try:
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except ImportError:
        device = "unknown"
    settings = {"experiment": setup.name, "split id (same in all experiments)": setup.split_id,
                "images train/val/test": "/".join(str(len(setup.part(s))) for s in ["train", "val", "test"]),
                "seed": cfg.SEED, "epochs": setup.epochs, "image size": cfg.IMAGE_SIZE, "batch size": cfg.BATCH_SIZE,
                "smoke test": setup.smoke, "training device": device}
    settings.update(more)
    return settings


def reuse_e1_yolo(setup, destination):
    """E2 and E3 contain the same 'nano, 640 px, default augmentation' run as E1. Instead of training it again,
    copy E1's weights (same recipe, same split). Returns True if copied."""
    suffix = "_smoke" if setup.smoke else ""
    source = cfg.EXPERIMENTS_DIR / "E1_baseline_vs_detector" / f"weights{suffix}" / "yolo"
    if (source / "best.pt").exists():
        shutil.copytree(source, destination, dirs_exist_ok=True)
        print("re-using E1 YOLO weights from", source)
        return True
    return False
