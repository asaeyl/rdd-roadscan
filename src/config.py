"""All settings of the project in ONE place.

Every experiment (E1, E2, E3) imports this file, so they all use the same seed, the same classes,
the same image size and the same split file.
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------- folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# THE frozen split. Created once by scripts/make_split.py, committed to GitHub, read by every experiment.
SPLIT_CSV = DATA_DIR / "splits_12000.csv"

# Big temporary files (copied images, YOLO training logs). NOT committed to GitHub.
WORK_DIR = Path(os.environ.get("RDD_WORK_DIR", PROJECT_ROOT / "work"))

# ----------------------------------------------------------------------------- data
SEED = 42                          # every random choice in the project uses this seed
TARGET_IMAGES = 12000               # size of the balanced subset
SHARE_WITHOUT_DAMAGE = 0.08        # about 8% of the subset = images with no damage at all
FRAMES_PER_BLOCK = 50              # frames with nearby numbers are near-identical -> they stay together
SPLIT_SHARES = {"train": 0.70, "val": 0.15, "test": 0.15}

DAMAGE_CLASSES = ["D00", "D10", "D20", "D40"]
CLASS_ID_TO_NAME = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}     # verified in notebooks/00_colab_runner.ipynb
HAS_COLUMNS = ["has_" + name for name in DAMAGE_CLASSES]        # has_D00, has_D10, ...

# ----------------------------------------------------------------------------- training (shared by all models)
IMAGE_SIZE = 640                   # input size in pixels
EPOCHS = 20                        # passes over the training images
BATCH_SIZE = 16                    # images per step
SAVE_LAST_WEIGHTS = True           # also keep last.pt (the final epoch), not only best.pt (best on validation)

# ----------------------------------------------------------------------------- baseline (ResNet-18)
BASELINE_LR_BACKBONE = 1e-4        # pretrained layers change slowly
BASELINE_LR_HEAD = 1e-3            # new last layer learns fast
BASELINE_WEIGHT_DECAY = 1e-4
BASELINE_MAX_POS_WEIGHT = 5        # cap for the class weight in the loss
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ----------------------------------------------------------------------------- detector (YOLO)
YOLO_MODEL = "yolov8n.pt"          # small COCO-pretrained detector
YOLO_TRAIN_EXTRA = {"close_mosaic": 5}       # settings used in EVERY YOLO run, so that runs stay comparable

# ----------------------------------------------------------------------------- timing
CPU_TIMING_IMAGES = 30             # how many test images are used to measure CPU time per image
