"""DETECTOR: YOLO training, evaluation and speed measurement (shared by E1, E2, E3)."""
import shutil
import time
from pathlib import Path

import numpy as np

from src import config as cfg


def train_yolo(run_name, start_weights, data_yaml, epochs, imgsz, batch, extra=None):
    """Train one YOLO model. `extra` = any additional Ultralytics settings (e.g. {"mosaic": 0}).
    Returns the folder where Ultralytics wrote its files."""
    from ultralytics import YOLO
    model = YOLO(start_weights)
    settings = {**cfg.YOLO_TRAIN_EXTRA, **(extra or {})}
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, seed=cfg.SEED,
                project=str(cfg.WORK_DIR / "runs"), name=run_name, exist_ok=True, **settings)
    return Path(model.trainer.save_dir)


def collect_weights(run_dir, dest_dir, save_last=cfg.SAVE_LAST_WEIGHTS):
    """Copy the small, important files of a training run into the repository folder:
    best.pt (best on validation), last.pt (final epoch), the training curves and the exact settings."""
    run_dir, dest_dir = Path(run_dir), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "weights" / "best.pt", dest_dir / "best.pt")
    if save_last and (run_dir / "weights" / "last.pt").exists():
        shutil.copy(run_dir / "weights" / "last.pt", dest_dir / "last.pt")
    for extra_file in ["results.csv", "results.png", "args.yaml"]:
        if (run_dir / extra_file).exists():
            shutil.copy(run_dir / extra_file, dest_dir / extra_file)
    return dest_dir


def evaluate_boxes(weights_path, data_yaml, imgsz, split="test"):
    """Box metrics on the chosen split: mAP@0.5, mAP@0.5:0.95, mean precision/recall and per-class AP@0.5."""
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, conf=0.001, verbose=False, plots=False,
                  project=str(cfg.WORK_DIR / "runs"), name="val_tmp", exist_ok=True)
    row = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
           "precision": float(m.box.mp), "recall": float(m.box.mr)}
    per_class = {name: np.nan for name in cfg.DAMAGE_CLASSES}
    for class_index, ap in zip(m.box.ap_class_index, m.box.ap50):
        per_class[cfg.DAMAGE_CLASSES[int(class_index)]] = float(ap)
    row.update({f"AP50_{name}": ap for name, ap in per_class.items()})
    return row


def image_level_scores(weights_path, image_paths, imgsz):
    """For each image: the highest YOLO confidence per damage type (0 if YOLO found none).
    This lets a detector answer the same question as the baseline: "is damage type X in this image?" """
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    scores = np.zeros((len(image_paths), len(cfg.DAMAGE_CLASSES)))
    for i, image_file in enumerate(image_paths):
        result = model.predict(str(image_file), imgsz=imgsz, conf=0.001, verbose=False)[0]
        for class_id, confidence in zip(result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
            scores[i, class_id] = max(scores[i, class_id], confidence)
    return scores


def measure_cpu_speed(weights_path, image_paths, imgsz):
    """Average time per image on the CPU in milliseconds (preprocess + model + postprocess), batch size 1."""
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    for image_file in image_paths[:3]:                                           # warm-up
        model.predict(str(image_file), imgsz=imgsz, device="cpu", verbose=False)
    pre, inf, post = [], [], []
    for image_file in image_paths:
        speed = model.predict(str(image_file), imgsz=imgsz, device="cpu", verbose=False)[0].speed
        pre.append(speed["preprocess"]); inf.append(speed["inference"]); post.append(speed["postprocess"])
    return {"cpu_ms_total": float(np.mean(pre) + np.mean(inf) + np.mean(post)), "cpu_ms_model_only": float(np.mean(inf))}


def model_size(weights_path):
    """Parameters in millions and weight file size in MB."""
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    return params, Path(weights_path).stat().st_size / 1e6
