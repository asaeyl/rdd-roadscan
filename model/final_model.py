"""FINAL MODEL = YOLO26s, 640 px, 20 epochs, trained again on the project split.

    python model/final_model.py --smoke        # 1 epoch on 5% of the data (output goes to model/final_model_smoke/): checks that everything works
    python model/final_model.py                # the real run: trains yolo26s for 20 epochs   (about as long as the E2 yolo26s_640 run)
    python model/final_model.py --reuse        # do NOT train again: only redo thresholds + evaluation with the saved model/final_model/best.pt

What it does
  1. trains yolo26s.pt (COCO-pretrained) at 640 px for 20 epochs on the TRAIN split (same split, seed and settings as E1-E3)
  2. picks ONE confidence threshold per damage type on the VALIDATION split (cost-aware F-beta; potholes weigh recall more)
  3. evaluates ONCE on the TEST split
  4. saves everything in model/final_model/ : best.pt, last.pt, artifact.json, results.xlsx/.txt, threshold curves, plots, training curves

Use it later (app, E5):   RoadDamageDetector().predict("photo.jpg")      (the class is defined in this file)
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src import config as cfg
from src import data, detector
from src.experiment import set_seed
from src.reporting import save_results

# ---------------------------------------------------------------- the final choice (change here only)
BASE_WEIGHTS = "yolo26s.pt"        # model: YOLO26 small, COCO-pretrained
IMAGE_SIZE = 640                   # input size in pixels (the size it is trained AND tested with)
EPOCHS = 20                        # passes over the training images
RUN_NAME = "final_yolo26s_640"

FINAL_MODEL_DIR = Path(__file__).resolve().parent / "final_model"       # model/final_model/   (E4, E5 and the app read this folder)

class sf:
    """Settings of the final model, E4 and E5 (all in one place, edit here)."""
    # cost of errors per damage type: F-beta with beta > 1 favours recall (a missed pothole is worse than a false alarm)
    BETA_PER_CLASS = {"D00": 1.0, "D10": 1.0, "D20": 1.0, "D40": 2.0}
    THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
    FALLBACK_THRESHOLD = 0.25            # used only if a damage type has no examples in the validation split
    IOU_THRESHOLD = 0.5                  # a detection is correct if its IoU with a real box is at least this
    CROP_PADDING = 0.10                  # extra border around each detected damage



# ============================================================================== thresholds
# ---------------------------------------------------------------- reading boxes
def load_ground_truth(label_path):
    """YOLO label file -> array (M, 5): class_id, x1, y1, x2, y2 (all relative, 0-1)."""
    rows = []
    for line in Path(label_path).read_text().splitlines():
        if line.strip():
            class_id, x, y, w, h = line.split()[:5]
            x, y, w, h = float(x), float(y), float(w), float(h)
            rows.append([int(class_id), x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    return np.array(rows, dtype=float).reshape(-1, 5)


def predict_boxes(weights_path, image_paths, imgsz, conf=0.001):
    """Run YOLO on images. Returns one array per image with rows: x1, y1, x2, y2 (relative), confidence, class_id."""
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    outputs = []
    for image_file in image_paths:
        boxes = model.predict(str(image_file), imgsz=imgsz, conf=conf, verbose=False)[0].boxes
        outputs.append(np.column_stack([boxes.xyxyn.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()]).reshape(-1, 6))
    return outputs


# ---------------------------------------------------------------- matching
def box_iou(a, b):
    """IoU between every box of a (N,4) and every box of b (M,4) -> (N,M)."""
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def match_predictions(predictions, ground_truths, n_classes=len(cfg.DAMAGE_CLASSES), iou_threshold=sf.IOU_THRESHOLD):
    """Mark every prediction as correct (True) or wrong (False), per damage type.
    predictions : list (one per image) of arrays (N, 6)   x1, y1, x2, y2, confidence, class
    ground_truths: list (one per image) of arrays (M, 5)  class, x1, y1, x2, y2
    Returns {class_id: (confidences, is_correct, number_of_real_boxes)}"""
    confs = {c: [] for c in range(n_classes)}
    flags = {c: [] for c in range(n_classes)}
    n_truth = {c: 0 for c in range(n_classes)}
    for preds, truths in zip(predictions, ground_truths):
        for c in range(n_classes):
            p = preds[preds[:, 5] == c]
            g = truths[truths[:, 0] == c][:, 1:5]
            n_truth[c] += len(g)
            if len(p) == 0:
                continue
            p = p[np.argsort(-p[:, 4])]                                   # most confident first
            ious = box_iou(p[:, :4], g) if len(g) else np.zeros((len(p), 0))
            used = np.zeros(len(g), dtype=bool)
            for i in range(len(p)):
                correct = False
                if len(g):
                    candidates = np.where(~used & (ious[i] >= iou_threshold), ious[i], -1.0)
                    best = int(candidates.argmax())
                    if candidates[best] >= 0:
                        used[best] = True
                        correct = True
                confs[c].append(p[i, 4])
                flags[c].append(correct)
    return {c: (np.array(confs[c]), np.array(flags[c], dtype=bool), n_truth[c]) for c in range(n_classes)}


# ---------------------------------------------------------------- curves and thresholds
def curve_from_matches(confidences, is_correct, n_truth, grid=sf.THRESHOLD_GRID, beta=1.0):
    """Precision, recall and F-beta for every threshold of the grid."""
    rows = []
    for t in grid:
        keep = confidences >= t
        tp = int(is_correct[keep].sum())
        fp = int(keep.sum()) - tp
        fn = n_truth - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / n_truth if n_truth else 0.0
        denominator = beta ** 2 * precision + recall
        f_beta = (1 + beta ** 2) * precision * recall / denominator if denominator else 0.0
        rows.append({"threshold": float(t), "precision": precision, "recall": recall, "f_beta": f_beta, "tp": tp, "fp": fp, "fn": fn})
    return pd.DataFrame(rows)


def choose_thresholds(matches, betas=sf.BETA_PER_CLASS, grid=sf.THRESHOLD_GRID):
    """Per damage type: the threshold with the best F-beta on this data (use VALIDATION data).
    Returns ({name: threshold}, {name: curve DataFrame})."""
    thresholds, curves = {}, {}
    for c, name in enumerate(cfg.DAMAGE_CLASSES):
        confidences, is_correct, n_truth = matches[c]
        curve = curve_from_matches(confidences, is_correct, n_truth, grid, betas[name])
        curves[name] = curve
        if n_truth == 0 or curve.f_beta.max() == 0:
            thresholds[name] = sf.FALLBACK_THRESHOLD                      # nothing to learn from
        else:
            best = np.flatnonzero(curve.f_beta.values == curve.f_beta.max())
            thresholds[name] = float(curve.threshold.values[best[len(best) // 2]])   # middle of the best plateau
    return thresholds, curves


def metrics_at_thresholds(matches, thresholds):
    """Precision / recall / F1 per damage type at the chosen thresholds (use on the TEST split)."""
    rows = {}
    for c, name in enumerate(cfg.DAMAGE_CLASSES):
        confidences, is_correct, n_truth = matches[c]
        row = curve_from_matches(confidences, is_correct, n_truth, [thresholds[name]], 1.0).iloc[0]
        rows[name] = {"threshold": thresholds[name], "precision": row.precision, "recall": row.recall,
                      "F1": 2 * row.precision * row.recall / (row.precision + row.recall) if row.precision + row.recall else 0.0,
                      "TP": int(row.tp), "FP": int(row.fp), "FN": int(row.fn), "real_boxes": n_truth}
    table = pd.DataFrame(rows).T
    table.loc["macro"] = table[["precision", "recall", "F1"]].astype(float).mean().to_dict() | {"threshold": np.nan, "TP": table.TP.sum(), "FP": table.FP.sum(), "FN": table.FN.sum(), "real_boxes": table.real_boxes.sum()}
    return table


# ============================================================================== the detector you load later (app, E5)
def to_pil(image):
    """Accept a file path, a PIL image or an RGB numpy array."""
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def crop_detections(image, detections, padding=sf.CROP_PADDING):
    """Cut every detected damage out of the image (with a small border). Returns [(crop, class_id), ...]."""
    image = to_pil(image)
    width, height = image.size
    crops = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
        box = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)), min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y)))
        if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
            crops.append((image.crop(box), d["class_id"]))
    return crops


class RoadDamageDetector:
    """YOLO weights + per-class confidence thresholds from model/final_model/artifact.json."""

    def __init__(self, folder=None):
        from ultralytics import YOLO
        self.folder = Path(folder or FINAL_MODEL_DIR)
        artifact_file = self.folder / "artifact.json"
        if not artifact_file.exists():
            raise FileNotFoundError(f"{artifact_file} not found. Create the final model first: python scripts/make_final_model.py")
        self.artifact = json.loads(artifact_file.read_text())
        self.image_size = self.artifact["image_size"]
        self.thresholds = self.artifact["confidence_thresholds"]          # {"D00": 0.31, ...}
        self.model = YOLO(str(self.folder / "best.pt"))

    def predict(self, image):
        """Detections above the per-class thresholds, most confident first.
        Each: class_id, class_name, confidence, box [x1, y1, x2, y2] in pixels, relative_area (box area / image area)."""
        pil = to_pil(image)
        width, height = pil.size
        result = self.model.predict(pil, imgsz=self.image_size, conf=min(self.thresholds.values()), verbose=False)[0]
        detections = []
        for box, conf, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                       result.boxes.cls.cpu().numpy().astype(int)):
            name = cfg.DAMAGE_CLASSES[class_id]
            if conf >= self.thresholds[name]:
                x1, y1, x2, y2 = (float(v) for v in box)
                detections.append({"class_id": int(class_id), "class_name": name, "confidence": float(conf),
                                   "box": [x1, y1, x2, y2], "relative_area": (x2 - x1) * (y2 - y1) / (width * height)})
        return sorted(detections, key=lambda d: -d["confidence"])



# ============================================================================== helpers
def evaluate_with_plots(best, data_yaml, imgsz, plots_dir):
    """Box metrics on TEST plus Ultralytics' own plots (confusion matrix, PR / F1 curves) copied to plots_dir."""
    from ultralytics import YOLO
    m = YOLO(str(best)).val(data=str(data_yaml), split="test", imgsz=imgsz, conf=0.001, plots=True, verbose=False,
                            project=str(cfg.WORK_DIR / "runs"), name="val_final", exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    for png in Path(m.save_dir).glob("*.png"):
        shutil.copy(png, plots_dir / png.name)
    row = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "precision": float(m.box.mp), "recall": float(m.box.mr)}
    per_class = {name: np.nan for name in cfg.DAMAGE_CLASSES}
    for class_index, ap in zip(m.box.ap_class_index, m.box.ap50):
        per_class[cfg.DAMAGE_CLASSES[int(class_index)]] = float(ap)
    row.update({f"AP50_{name}": ap for name, ap in per_class.items()})
    return row


def plot_threshold_curves(curves, thresholds, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, name in zip(axes.ravel(), cfg.DAMAGE_CLASSES):
        c = curves[name]
        ax.plot(c.threshold, c.precision, label="precision"); ax.plot(c.threshold, c.recall, label="recall")
        ax.plot(c.threshold, c.f_beta, label=f"F-beta (beta={sf.BETA_PER_CLASS[name]:g})", linewidth=2)
        ax.axvline(thresholds[name], color="k", linestyle="--", label=f"chosen {thresholds[name]:.2f}")
        ax.set_title(name); ax.set_xlabel("confidence threshold"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.suptitle("Validation split: how the confidence thresholds were chosen"); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def paths_for(split_table, split):
    """(image files, label files) of the YOLO folder for one split."""
    names = split_table[split_table.split == split].name
    yolo = cfg.WORK_DIR / "yolo"
    return [str(yolo / "images" / split / f"{n}.jpg") for n in names], [yolo / "labels" / split / f"{n}.txt" for n in names]


def read_run_args(run_folder):
    """Settings Ultralytics saved next to the weights (args.yaml): model name, epochs, ... ({} if not there)."""
    try:
        import yaml
        return yaml.safe_load((Path(run_folder) / "args.yaml").read_text()) or {}
    except Exception:
        return {}



# ============================================================================== create the final model
def main():
    parser = argparse.ArgumentParser(description="Train and save the final model (yolo26s, 640 px, 20 epochs)")
    parser.add_argument("--smoke", action="store_true", help="quick test: 1 epoch on 5%% of the data, output in model/final_model_smoke/")
    parser.add_argument("--reuse", action="store_true", help="do not train: use the saved model/final_model/best.pt and redo thresholds + evaluation")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"default {EPOCHS}")
    args = parser.parse_args()
    if args.smoke and args.reuse:
        sys.exit("--smoke and --reuse cannot be combined")

    epochs = 1 if args.smoke else args.epochs
    final_dir = FINAL_MODEL_DIR.parent / ("final_model_smoke" if args.smoke else "final_model")
    final_dir.mkdir(parents=True, exist_ok=True)

    set_seed()
    split_table = data.attach_paths(data.load_split_csv(), data.get_dataset_dir())
    data_yaml = data.build_yolo_dataset(split_table)
    split_id = data.split_fingerprint(split_table)
    print(f"== final model | {BASE_WEIGHTS} | {IMAGE_SIZE} px | {epochs} epochs | split id {split_id}{' | SMOKE TEST' if args.smoke else ''}")

    # ---- 1) train (or re-use the saved weights)
    if args.reuse:
        best = final_dir / "best.pt"
        if not best.exists():
            sys.exit(f"{best} not found - run without --reuse once to train the model")
        run_args = read_run_args(final_dir / "training")
        print("re-using", best)
    else:
        run_dir = detector.train_yolo(RUN_NAME, BASE_WEIGHTS, data_yaml, epochs, IMAGE_SIZE, cfg.BATCH_SIZE,
                                      {"fraction": 0.05} if args.smoke else None)
        trained = detector.collect_weights(run_dir, cfg.WORK_DIR / "final_tmp", save_last=cfg.SAVE_LAST_WEIGHTS)
        best = trained / "best.pt"
        run_args = read_run_args(trained)
        shutil.copy(best, final_dir / "best.pt")
        if (trained / "last.pt").exists():
            shutil.copy(trained / "last.pt", final_dir / "last.pt")                # the weights of the final epoch
        (final_dir / "training").mkdir(exist_ok=True)                              # training curves + exact training settings
        for name in ["results.csv", "results.png", "args.yaml"]:
            if (trained / name).exists():
                shutil.copy(trained / name, final_dir / "training" / name)

    # ---- 2) thresholds on VALIDATION
    val_images, val_labels = paths_for(split_table, "val")
    matches_val = match_predictions(predict_boxes(best, val_images, IMAGE_SIZE), [load_ground_truth(p) for p in val_labels])
    thresholds, curves = choose_thresholds(matches_val)
    print("confidence thresholds (chosen on validation):", thresholds)

    # ---- 3) report on TEST, once
    test_images, test_labels = paths_for(split_table, "test")
    matches_test = match_predictions(predict_boxes(best, test_images, IMAGE_SIZE), [load_ground_truth(p) for p in test_labels])
    test_table = metrics_at_thresholds(matches_test, thresholds)
    box = evaluate_with_plots(best, data_yaml, IMAGE_SIZE, final_dir / "plots")

    # ---- 4) write model/final_model/
    artifact = {"model": BASE_WEIGHTS, "source": f"trained here: {BASE_WEIGHTS}, {IMAGE_SIZE} px, {epochs} epochs" if not args.reuse else "re-used saved weights",
                "image_size": IMAGE_SIZE, "epochs": run_args.get("epochs", epochs), "optimizer": run_args.get("optimizer"),
                "classes": cfg.DAMAGE_CLASSES, "confidence_thresholds": thresholds, "beta_per_class": sf.BETA_PER_CLASS,
                "thresholds_chosen_on": "validation split", "iou_threshold": sf.IOU_THRESHOLD,
                "test_mAP50": box["mAP50"], "test_mAP50-95": box["mAP50-95"], "split_id": split_id, "seed": cfg.SEED,
                "smoke_test": args.smoke, "created": datetime.now().strftime("%Y-%m-%d %H:%M")}
    (final_dir / "artifact.json").write_text(json.dumps(artifact, indent=2))
    pd.concat({n: c for n, c in curves.items()}, names=["class"]).reset_index(level=0).to_csv(final_dir / "thresholds_validation.csv", index=False)
    plot_threshold_curves(curves, thresholds, final_dir / "f1_vs_confidence.png")
    notes = ["Trained on the TRAIN split only. Thresholds were chosen on the VALIDATION split; the TEST split was used once, for this table.",
             "Cost-aware choice: F-beta per damage type; beta=2 for potholes (D40) means recall counts more than precision.",
             "Precision/recall/F1 below are at the chosen thresholds (IoU >= 0.5); mAP is threshold-free.",
             "best.pt = epoch with the best validation score; last.pt = final epoch. artifact.json belongs to best.pt."]
    save_results(final_dir, "Final model - yolo26s, 640 px", {"test metrics at the chosen thresholds": test_table, "test box metrics (mAP)": pd.DataFrame([box], index=["final model"])},
                 {**{k: v for k, v in artifact.items() if k != "confidence_thresholds"}, "confidence_thresholds": thresholds}, notes)
    (final_dir / "README.md").write_text(
        "# Final model\n\nYOLO26s, 640 px, trained for 20 epochs on the project split. Weights `best.pt` + `artifact.json` "
        "(confidence thresholds per damage type, image size, split id).\n\n"
        "Load it from Python:\n```python\nimport importlib.util\nspec = importlib.util.spec_from_file_location('final_model', 'model/final_model.py')\n"
        "fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)\ndetector = fm.RoadDamageDetector()\nprint(detector.predict('photo.jpg'))\n```\n\n"
        f"Thresholds (validation): {thresholds}\n\nTest mAP@0.5 = {box['mAP50']:.3f}, mAP@0.5:0.95 = {box['mAP50-95']:.3f}. "
        "Details: `results.txt`, `results.xlsx`, `f1_vs_confidence.png`, `plots/`, `training/`.\n")
    print(test_table.round(3)); print("\nFinal model saved in", final_dir)


if __name__ == "__main__":
    main()
