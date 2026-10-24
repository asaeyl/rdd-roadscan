"""FINE-TUNED FINAL MODEL = the final model (model/final_model/best.pt) trained further on OUR local photos, with the LOCAL classes.

python fine-tuning/final_model.py --local-dir /content/local_photos --check-only  # BEFORE training: are the labels sane and do they fit the model?
python fine-tuning/final_model.py --local-dir /content/local_photos --smoke      # 1 epoch on 5% of the data: checks that everything works
python fine-tuning/final_model.py --local-dir /content/local_photos              # the real run: 20 epochs, same settings as model/final_model.py
python fine-tuning/final_model.py --local-dir /content/local_photos --reuse      # do NOT train again: redo thresholds + evaluation only

THE CLASSES ARE DIFFERENT. The local photos (Roboflow project 'saudird') are labelled with 3 classes:
    0 = Crack, 1 = Pothole, 2 = Rut          (see LOCAL_CLASSES below)
The final model has 4 classes: D00, D10, D20 (three crack types), D40 (pothole). They do NOT match id for id, so the labels are used as they
are and the fine-tuned model gets 3 outputs. Ultralytics keeps the weights of every layer whose shape still fits (backbone, neck) and starts
the class outputs new. The fine-tuned model is therefore a 3-class model: the app / E4 code must use LOCAL_CLASSES for it, not D00-D40.

Everything else is taken FROM model/final_model.py (it is imported, not copied): model size, image size (640), epochs (20), batch size,
seed, Ultralytics settings, IoU, the confidence-threshold grid. The cost-aware threshold choice is the same (beta 2 for potholes).

The local photos (820 at the moment) are split ONCE into train / validation / test (default 70 / 15 / 15 = about 574 / 123 / 123) by SOURCE
photo (Roboflow copies of one photo stay together), and the split is chosen so that validation and test contain enough boxes of every class.
It is saved in <output>/local_split.csv: the next experiment (E4 on the fine-tuned model) must use ONLY the 'test' photos.

What it does
  1. checks the photos and labels, splits them, builds a YOLO folder in the work directory (not in the repository)
  2. fine-tunes best.pt on the local TRAIN photos (20 epochs, validation on the local VALIDATION photos)
  3. picks ONE confidence threshold per local class on the VALIDATION photos (same cost-aware F-beta as the final model)
  4. evaluates ONCE on the local TEST photos, next to the ORIGINAL final model on the same photos (its D00/D10/D20 boxes merged into Crack,
     D40 = Pothole; it has no Rut class, so Rut can only be missed by it)
  5. saves everything in fine-tuning/final_model/ : best.pt, last.pt, artifact.json, results, plots
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src import config as cfg
from src.experiment import set_seed
from src.reporting import save_results

FINAL_MODEL_FILE = PROJECT_ROOT / "model" / "final_model.py"
ORIGINAL_DIR = PROJECT_ROOT / "model" / "final_model"                    # the model we start from
FINE_TUNED_DIR = Path(__file__).resolve().parent / "final_model"         # fine-tuning/final_model/ (output)
RUN_NAME = "finetune_local_yolo26s_640"
LOCAL_DIR_ENV = "RDD_LOCAL_DIR"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------- the local classes (edit here only)
LOCAL_CLASSES = ["Crack", "Pothole", "Rut"]                              # label ids 0, 1, 2 in the local labels (data.yaml of the Roboflow export)
LOCAL_BETA = {"Crack": 1.0, "Pothole": 2.0, "Rut": 1.0}                  # as in the final model: a missed pothole is worse than a false alarm
ORIGINAL_TO_LOCAL = {"D00": "Crack", "D10": "Crack", "D20": "Crack", "D40": "Pothole"}      # the original model has nothing for Rut
MIN_BOXES_PER_CLASS = 15                                                 # the split tries to give validation and test at least this many boxes per class
_FINAL = None


# ============================================================================== loading things
def get_final():
    """model/final_model.py loaded as a module: its settings and generic functions are re-used, so nothing can drift apart."""
    global _FINAL
    if _FINAL is None:
        if not FINAL_MODEL_FILE.exists():
            sys.exit(f"{FINAL_MODEL_FILE} not found.")
        spec = importlib.util.spec_from_file_location("final_model_file", FINAL_MODEL_FILE)
        _FINAL = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_FINAL)
    return _FINAL


def local_dir(cli_value=""):
    value = cli_value or os.environ.get(LOCAL_DIR_ENV, "")
    return Path(value) if value else None


def check_model_classes(weights):
    """The ORIGINAL weights must have the classes D00, D10, D20, D40 in this order (the mapping to Crack / Pothole relies on it)."""
    from ultralytics import YOLO
    names = {int(k): v for k, v in YOLO(str(weights)).names.items()}
    expected = {i: n for i, n in enumerate(cfg.DAMAGE_CLASSES)}
    print("class names inside the starting weights:", names)
    if names != expected:
        sys.exit(f"The starting weights use {names} but the project uses {expected}.")


# ============================================================================== the local photos
def check_local_data(folder):
    """Every photo needs a label file (EMPTY = no damage). Returns {photo name: (image path, label path)}."""
    images, labels = Path(folder) / "images", Path(folder) / "labels"
    if not images.is_dir() or not labels.is_dir():
        sys.exit(f"{images} and {labels} are needed.")
    photos = sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        sys.exit(f"no photos (.jpg/.png) in {images}")
    problems, found = [], {}
    for photo in photos:
        label_file = labels / (photo.stem + ".txt")
        if not label_file.exists():
            problems.append(f"no label file for {photo.name} (create an EMPTY {photo.stem}.txt if the photo shows no damage)")
            continue
        for number, line in enumerate(label_file.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parts = line.split()
                assert len(parts) == 5, "need exactly 5 numbers (polygon / segmentation labels are not supported)"
                class_id, values = int(parts[0]), [float(v) for v in parts[1:5]]
                assert class_id in range(len(LOCAL_CLASSES)), f"class id must be 0-{len(LOCAL_CLASSES) - 1} ({', '.join(LOCAL_CLASSES)})"
                assert all(0 <= v <= 1 for v in values) and values[2] > 0 and values[3] > 0, "values must be 0-1 and width / height > 0"
            except (ValueError, AssertionError, IndexError) as error:
                problems.append(f"bad line {number} in {label_file.name}: '{line[:60]}' ({error})")
        found[photo.name] = (photo, label_file)
    if problems:
        sys.exit("Problems with the local photos:\n  - " + "\n  - ".join(problems[:15]) + ("\n  ... and more" if len(problems) > 15 else ""))
    return found


def photo_box_counts(photos):
    """{photo name: array with the number of boxes per local class}."""
    counts = {}
    for name, (_, label_file) in photos.items():
        row = np.zeros(len(LOCAL_CLASSES), dtype=int)
        for line in label_file.read_text().splitlines():
            if line.strip():
                row[int(line.split()[0])] += 1
        counts[name] = row
    return counts


def source_key(name):
    """Roboflow names a photo '<source>_jpg.rf.<hash>': copies of one source photo share the part before '_jpg.rf.'."""
    return Path(name).stem.split("_jpg.rf.")[0]


def make_split(counts, val_share, test_share, block_size, seed):
    """train / val / test by SOURCE photo (+ blocks of block_size consecutive sources). Tries up to 300 seeded shuffles and keeps the first one in which
    validation AND test have at least MIN_BOXES_PER_CLASS boxes of every class (else the best one)."""
    names = sorted(counts)
    groups = {}
    for name in names:
        groups.setdefault(source_key(name), []).append(name)
    keys = sorted(groups)
    blocks = [keys[i:i + block_size] for i in range(0, len(keys), block_size)]
    target_test, target_val = round(len(names) * test_share), round(len(names) * val_share)
    best = None
    for attempt in range(300):
        order = np.random.default_rng(seed + attempt).permutation(len(blocks))
        assignment, filled = {}, {"test": 0, "val": 0}
        for index in order:
            split = "test" if filled["test"] < target_test else ("val" if filled["val"] < target_val else "train")
            members = [n for key in blocks[index] for n in groups[key]]
            for n in members:
                assignment[n] = split
            if split in filled:
                filled[split] += len(members)
        score = min(int(sum(counts[n] for n in names if assignment[n] == s).min()) for s in ("val", "test"))
        if best is None or score > best[0]:
            best = (score, assignment)
        if score >= MIN_BOXES_PER_CLASS:
            break
    score, assignment = best
    table = pd.DataFrame({"name": names, "split": [assignment[n] for n in names]})
    if min((table.split == s).sum() for s in ("train", "val", "test")) == 0:
        sys.exit("A split came out empty: use more photos or other shares.")
    if score < MIN_BOXES_PER_CLASS:
        print(f"WARNING: in the best split validation / test have only {score} boxes of the rarest class (wanted {MIN_BOXES_PER_CLASS}): "
              "the numbers of that class will be noisy.")
    return table


def count_table(split_table, counts):
    """Photos, photos without damage and boxes per split and local class."""
    rows = {}
    for split in ("train", "val", "test"):
        names = list(split_table[split_table.split == split].name)
        total = sum(counts[n] for n in names) if names else np.zeros(len(LOCAL_CLASSES), dtype=int)
        rows[split] = {"photos": len(names), "no damage": sum(1 for n in names if counts[n].sum() == 0), **{c: int(v) for c, v in zip(LOCAL_CLASSES, total)}}
    return pd.DataFrame(rows).T


def build_yolo_local(split_table, photos):
    """Copy the photos into <work dir>/yolo_local/{images,labels}/{train,val,test} and write data.yaml. Returns the yaml path."""
    import yaml
    root = cfg.WORK_DIR / "yolo_local"
    if root.exists():
        shutil.rmtree(root)                                              # also removes old Ultralytics label caches
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True); (root / "labels" / split).mkdir(parents=True)
    for name, split in zip(split_table.name, split_table.split):
        image_file, label_file = photos[name]
        shutil.copy2(image_file, root / "images" / split / image_file.name)
        shutil.copy2(label_file, root / "labels" / split / (image_file.stem + ".txt"))
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
                                         "names": {i: n for i, n in enumerate(LOCAL_CLASSES)}}, sort_keys=False))
    return data_yaml


def paths_for(split_table, split):
    """(image files, label files) of the local YOLO folder for one split."""
    root = cfg.WORK_DIR / "yolo_local"
    names = list(split_table[split_table.split == split].name)
    return ([str(root / "images" / split / n) for n in names], [root / "labels" / split / (Path(n).stem + ".txt") for n in names])


def split_fingerprint(split_table):
    return hashlib.sha1(split_table.sort_values("name").to_csv(index=False).encode()).hexdigest()[:10]


# ============================================================================== thresholds and metrics for the local classes
def choose_thresholds(fm, matches):
    """Per local class: the threshold with the best F-beta on the validation photos (same rule as model/final_model.py)."""
    thresholds, curves = {}, {}
    for c, name in enumerate(LOCAL_CLASSES):
        confidences, is_correct, n_truth = matches[c]
        curve = fm.curve_from_matches(confidences, is_correct, n_truth, fm.sf.THRESHOLD_GRID, LOCAL_BETA[name])
        curves[name] = curve
        if n_truth == 0 or curve.f_beta.max() == 0:
            thresholds[name] = fm.sf.FALLBACK_THRESHOLD
        else:
            best = np.flatnonzero(curve.f_beta.values == curve.f_beta.max())
            thresholds[name] = float(curve.threshold.values[best[len(best) // 2]])      # middle of the best plateau
    return thresholds, curves


def metrics_at(fm, matches, thresholds):
    """Precision / recall / F1 per local class at the given thresholds, plus a macro row over all classes and one over Crack + Pothole."""
    rows = {}
    for c, name in enumerate(LOCAL_CLASSES):
        confidences, is_correct, n_truth = matches[c]
        r = fm.curve_from_matches(confidences, is_correct, n_truth, [thresholds[name]], 1.0).iloc[0]
        rows[name] = {"threshold": thresholds[name], "precision": r.precision, "recall": r.recall,
                      "F1": 2 * r.precision * r.recall / (r.precision + r.recall) if r.precision + r.recall else 0.0,
                      "TP": int(r.tp), "FP": int(r.fp), "FN": int(r.fn), "real_boxes": n_truth}
    table = pd.DataFrame(rows).T
    for label, subset in (("macro (all classes)", list(LOCAL_CLASSES)), ("macro (Crack + Pothole)", ["Crack", "Pothole"])):
        table.loc[label] = table.loc[subset, ["precision", "recall", "F1"]].astype(float).mean().to_dict() | {
            "threshold": np.nan, "TP": table.loc[subset, "TP"].sum(), "FP": table.loc[subset, "FP"].sum(),
            "FN": table.loc[subset, "FN"].sum(), "real_boxes": table.loc[subset, "real_boxes"].sum()}
    return table


def run_matches(fm, weights, image_size, split_table, split):
    """Run a model (its own class ids = local classes) on one local split and match its boxes to the real ones."""
    images, labels = paths_for(split_table, split)
    return fm.match_predictions(fm.predict_boxes(weights, images, image_size), [fm.load_ground_truth(p) for p in labels], n_classes=len(LOCAL_CLASSES))


def nms_per_class(fm, boxes, iou=0.5):
    """Greedy NMS per class (merging D00 / D10 / D20 into Crack can create duplicate boxes on the same crack)."""
    keep = []
    for c in np.unique(boxes[:, 5]):
        group = boxes[boxes[:, 5] == c]
        group = group[np.argsort(-group[:, 4])]
        while len(group):
            keep.append(group[0])
            if len(group) == 1:
                break
            group = group[1:][fm.box_iou(group[:1, :4], group[1:, :4])[0] < iou]
    return np.array(keep).reshape(-1, 6)


def original_matches(fm, split_table, split, original_artifact):
    """The ORIGINAL model on a local split: its own per-class thresholds, then D00 / D10 / D20 -> Crack (id 0), D40 -> Pothole (id 1), NMS."""
    images, labels = paths_for(split_table, split)
    thresholds = original_artifact["confidence_thresholds"]
    mapped = []
    for boxes in fm.predict_boxes(ORIGINAL_DIR / "best.pt", images, original_artifact["image_size"]):
        if len(boxes):
            need = np.array([thresholds[cfg.DAMAGE_CLASSES[int(c)]] for c in boxes[:, 5]])
            boxes = boxes[boxes[:, 4] >= need].copy()
            boxes[:, 5] = [LOCAL_CLASSES.index(ORIGINAL_TO_LOCAL[cfg.DAMAGE_CLASSES[int(c)]]) for c in boxes[:, 5]]
        mapped.append(nms_per_class(fm, boxes) if len(boxes) else np.zeros((0, 6)))
    return fm.match_predictions(mapped, [fm.load_ground_truth(p) for p in labels], n_classes=len(LOCAL_CLASSES))


def compare_tables(original, tuned):
    columns = ["precision", "recall", "F1"]
    out = pd.concat({"original (merged)": original[columns].astype(float), "fine-tuned": tuned[columns].astype(float)}, axis=1)
    out.columns = [f"{model} {metric}" for model, metric in out.columns]
    for metric in columns:
        out[f"{metric} change"] = out[f"fine-tuned {metric}"] - out[f"original (merged) {metric}"]
    return out


def evaluate_with_plots(best, data_yaml, imgsz, plots_dir):
    """Box metrics on the local TEST split plus Ultralytics' own plots (confusion matrix, PR / F1 curves)."""
    from ultralytics import YOLO
    m = YOLO(str(best)).val(data=str(data_yaml), split="test", imgsz=imgsz, conf=0.001, plots=True, verbose=False,
                            project=str(cfg.WORK_DIR / "runs"), name="val_finetune", exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    for png in Path(m.save_dir).glob("*.png"):
        shutil.copy(png, plots_dir / png.name)
    row = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)}
    per_class = {name: np.nan for name in LOCAL_CLASSES}
    for class_index, ap in zip(m.box.ap_class_index, m.box.ap50):
        per_class[LOCAL_CLASSES[int(class_index)]] = float(ap)
    row.update({f"AP50_{name}": ap for name, ap in per_class.items()})
    return row


def plot_threshold_curves(curves, thresholds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(LOCAL_CLASSES), figsize=(5 * len(LOCAL_CLASSES), 4))
    for ax, name in zip(np.atleast_1d(axes), LOCAL_CLASSES):
        c = curves[name]
        ax.plot(c.threshold, c.precision, label="precision"); ax.plot(c.threshold, c.recall, label="recall")
        ax.plot(c.threshold, c.f_beta, label=f"F-beta (beta={LOCAL_BETA[name]:g})", linewidth=2)
        ax.axvline(thresholds[name], color="k", linestyle="--", label=f"chosen {thresholds[name]:.2f}")
        ax.set_title(name); ax.set_xlabel("confidence threshold"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.suptitle("Local validation photos: how the confidence thresholds were chosen"); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


# ============================================================================== do the labels fit the original model?
def check_labels(fm, weights, photos, out_dir, min_conf=0.25, samples_per_class=6):
    """1) boxes / photos per local class  2) what the ORIGINAL model (confidence >= min_conf) sees at every labelled box (IoU >= 0.5):
    Crack should be seen mostly as D00 / D10 / D20, Pothole as D40, Rut as something or nothing  3) cropped label boxes per class to look at."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    names = sorted(photos)
    predictions = fm.predict_boxes(weights, [str(photos[n][0]) for n in names], fm.IMAGE_SIZE)
    rows, boxes_by_class = [], {c: [] for c in LOCAL_CLASSES}
    for name, pred in zip(names, predictions):
        pred = pred[pred[:, 4] >= min_conf]
        for row in fm.load_ground_truth(photos[name][1]):
            label = LOCAL_CLASSES[int(row[0])]
            seen = "no model box"
            if len(pred):
                ious = fm.box_iou(row[None, 1:5], pred[:, :4])[0]
                best = int(ious.argmax())
                if ious[best] >= 0.5:
                    seen = cfg.DAMAGE_CLASSES[int(pred[best, 5])]
            rows.append({"photo": name, "label": label, "model sees": seen})
            boxes_by_class[label].append((name, row[1:5]))
    table = pd.DataFrame(rows)
    if table.empty:
        sys.exit("no labelled boxes found in the local photos.")
    columns = cfg.DAMAGE_CLASSES + ["no model box"]
    confusion = pd.crosstab(table["label"], table["model sees"]).reindex(index=LOCAL_CLASSES, columns=columns, fill_value=0)
    summary = pd.DataFrame({"boxes": confusion.sum(axis=1), "photos": [table[table.label == c].photo.nunique() for c in LOCAL_CLASSES],
                            "model sees nothing": confusion["no model box"]})
    print(f"\nlabelled boxes per local class and what the ORIGINAL model sees there (confidence >= {min_conf:.2f}):")
    print(summary.to_string()); print("\nrows = your label, columns = class of the original model's box that overlaps it:"); print(confusion.to_string())
    for local_class in LOCAL_CLASSES:
        expected = {d for d, l in ORIGINAL_TO_LOCAL.items() if l == local_class}
        matched = confusion.loc[local_class, cfg.DAMAGE_CLASSES]
        if not expected:
            print(f"note: {local_class} has no equivalent in the original model; the model sees it as: " + ", ".join(f"{k} {int(v)}" for k, v in matched.items() if v))
        elif matched.sum() >= 5 and matched[list(expected)].sum() / matched.sum() < 0.5:
            print(f"WARNING: {local_class} boxes are mostly seen by the original model as {matched.idxmax()} (expected {sorted(expected)}). "
                  "Check that the label ids really are 0 = Crack, 1 = Pothole, 2 = Rut.")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.SEED)
    fig, axes = plt.subplots(len(LOCAL_CLASSES), samples_per_class, figsize=(2.6 * samples_per_class, 2.6 * len(LOCAL_CLASSES)))
    for r, c in enumerate(LOCAL_CLASSES):
        pool = boxes_by_class[c]
        chosen = [pool[i] for i in rng.permutation(len(pool))[:samples_per_class]]
        for k in range(samples_per_class):
            ax = axes[r, k]; ax.axis("off")
            if k < len(chosen):
                photo_name, (x1, y1, x2, y2) = chosen[k]
                image = ImageOps.exif_transpose(Image.open(photos[photo_name][0])).convert("RGB")
                W, H = image.size
                ax.imshow(image.crop((int(x1 * W), int(y1 * H), max(int(x2 * W), int(x1 * W) + 2), max(int(y2 * H), int(y1 * H) + 2))))
            if k == 0:
                ax.set_title(f"your label {c}", fontsize=10, loc="left")
    plt.tight_layout(); fig.savefig(out_dir / "label_samples.png", dpi=80); plt.close(fig)
    table_dir = FINE_TUNED_DIR.parent / "label_check"                       # numbers only (no photos) go into the repository
    table_dir.mkdir(exist_ok=True)
    summary.to_csv(table_dir / "label_check_summary.csv"); confusion.to_csv(table_dir / "label_check_confusion.csv")
    print("\nlook at:", out_dir / "label_samples.png", "(every row should show ONE kind of damage, the one its title names)")
    print("numbers:", table_dir)


# ============================================================================== fine-tune
def main():
    fm = get_final()
    parser = argparse.ArgumentParser(description="Fine-tune the final model on the local photos (3 local classes, same settings as model/final_model.py)")
    parser.add_argument("--local-dir", default="", help=f"folder with images/ and labels/; or set {LOCAL_DIR_ENV}")
    parser.add_argument("--base-weights", default="", help="weights to start from (default: model/final_model/best.pt)")
    parser.add_argument("--check-only", action="store_true", help="only check the labels against the original model (no training), then stop")
    parser.add_argument("--out-dir", default="", help="where the label-check pictures go (default: <local-dir>/label_check_output)")
    parser.add_argument("--smoke", action="store_true", help="quick test: 1 epoch on 5%% of the data, output in fine-tuning/final_model_smoke/")
    parser.add_argument("--reuse", action="store_true", help="do not train: use the saved fine-tuning/final_model/best.pt, redo thresholds + evaluation")
    parser.add_argument("--epochs", type=int, default=fm.EPOCHS, help=f"default {fm.EPOCHS} (as in model/final_model.py)")
    parser.add_argument("--val-share", type=float, default=0.15, help="share of the local photos for validation (default 0.15)")
    parser.add_argument("--test-share", type=float, default=0.15, help="share of the local photos for the final test (default 0.15)")
    parser.add_argument("--block-size", type=int, default=1, help="source photos that stay together in the same split (default 1 = each source photo on its own)")
    args = parser.parse_args()
    if args.smoke and args.reuse:
        sys.exit("--smoke and --reuse cannot be combined")

    folder = local_dir(args.local_dir)
    if folder is None:
        sys.exit(f"Give the folder with --local-dir or set the environment variable {LOCAL_DIR_ENV}")
    base = Path(args.base_weights) if args.base_weights else ORIGINAL_DIR / "best.pt"
    original_artifact_file = ORIGINAL_DIR / "artifact.json"
    if not base.exists() or not original_artifact_file.exists():
        sys.exit(f"{base} or {original_artifact_file} not found. In Colab run the 'Download the project' cell first (the saved final model).")
    original_artifact = json.loads(original_artifact_file.read_text())

    photos = check_local_data(folder)
    check_model_classes(base)
    if args.check_only:
        out_dir = Path(args.out_dir) if args.out_dir else folder / "label_check_output"
        if PROJECT_ROOT in out_dir.resolve().parents:
            sys.exit(f"{out_dir} is inside the repository, which is public. Choose another folder for --out-dir.")
        check_labels(fm, base, photos, out_dir)
        return

    epochs = 1 if args.smoke else args.epochs
    final_dir = FINE_TUNED_DIR.parent / ("final_model_smoke" if args.smoke else "final_model")
    final_dir.mkdir(parents=True, exist_ok=True)
    set_seed()

    # ---- 0) split the local photos, build the YOLO folder
    counts = photo_box_counts(photos)
    split_file = final_dir / "local_split.csv"
    if args.reuse and split_file.exists():
        split_table = pd.read_csv(split_file)
        if set(split_table.name) != set(photos):
            sys.exit("The photos changed since the saved split (local_split.csv). Run without --reuse.")
    else:
        split_table = make_split(counts, args.val_share, args.test_share, args.block_size, cfg.SEED)
    split_table.to_csv(split_file, index=False)
    data_yaml = build_yolo_local(split_table, photos)
    split_id = split_fingerprint(split_table)
    box_counts = count_table(split_table, counts)
    print(f"== fine-tuning | start: {base} | {fm.BASE_WEIGHTS} | {fm.IMAGE_SIZE} px | {epochs} epochs | local photos {len(photos)} | "
          f"classes {LOCAL_CLASSES} | split id {split_id}{' | SMOKE TEST' if args.smoke else ''}")
    print(box_counts.to_string())
    for split in ("val", "test"):
        thin = [c for c in LOCAL_CLASSES if box_counts.loc[split, c] < MIN_BOXES_PER_CLASS]
        if thin:
            print(f"WARNING: the local {split} split has fewer than {MIN_BOXES_PER_CLASS} boxes of {', '.join(thin)}: "
                  + ("their thresholds are unreliable." if split == "val" else "their test numbers are noisy."))

    # ---- 1) train (or re-use the saved weights)
    if args.reuse:
        best = final_dir / "best.pt"
        if not best.exists():
            sys.exit(f"{best} not found - run without --reuse once to train the model")
        run_args = fm.read_run_args(final_dir / "training")
        print("re-using", best)
    else:
        run_dir = fm.detector.train_yolo(RUN_NAME, str(base), data_yaml, epochs, fm.IMAGE_SIZE, cfg.BATCH_SIZE,
                                         {"fraction": 0.05} if args.smoke else None)
        trained = fm.detector.collect_weights(run_dir, cfg.WORK_DIR / "finetune_tmp", save_last=cfg.SAVE_LAST_WEIGHTS)
        best = trained / "best.pt"
        run_args = fm.read_run_args(trained)
        shutil.copy(best, final_dir / "best.pt")
        if (trained / "last.pt").exists():
            shutil.copy(trained / "last.pt", final_dir / "last.pt")
        (final_dir / "training").mkdir(exist_ok=True)
        for name in ["results.csv", "results.png", "args.yaml"]:
            if (trained / name).exists():
                shutil.copy(trained / name, final_dir / "training" / name)

    # ---- 2) thresholds on the local VALIDATION photos
    thresholds, curves = choose_thresholds(fm, run_matches(fm, best, fm.IMAGE_SIZE, split_table, "val"))
    print("confidence thresholds (chosen on the local validation photos):", thresholds)

    # ---- 3) the local TEST photos, once: fine-tuned model and (for comparison) the original final model on the same photos
    tuned_table = metrics_at(fm, run_matches(fm, best, fm.IMAGE_SIZE, split_table, "test"), thresholds)
    original_table = metrics_at(fm, original_matches(fm, split_table, "test", original_artifact), {c: 0.0 for c in LOCAL_CLASSES})   # already thresholded
    comparison = compare_tables(original_table, tuned_table)
    box = evaluate_with_plots(best, data_yaml, fm.IMAGE_SIZE, final_dir / "plots")

    # ---- 4) write fine-tuning/final_model/
    artifact = {"model": fm.BASE_WEIGHTS, "source": f"fine-tuned from {base.as_posix()}: {fm.IMAGE_SIZE} px, {epochs} epochs, on local photos"
                if not args.reuse else "re-used saved fine-tuned weights",
                "fine_tuned_from": base.as_posix(), "image_size": fm.IMAGE_SIZE, "epochs": run_args.get("epochs", epochs),
                "optimizer": run_args.get("optimizer"), "classes": LOCAL_CLASSES, "confidence_thresholds": thresholds,
                "beta_per_class": LOCAL_BETA, "thresholds_chosen_on": "local validation photos", "iou_threshold": fm.sf.IOU_THRESHOLD,
                "original_to_local_classes": ORIGINAL_TO_LOCAL, "local_test_mAP50": box["mAP50"], "local_test_mAP50-95": box["mAP50-95"],
                "local_photos": {s: int(box_counts.loc[s, "photos"]) for s in ("train", "val", "test")},
                "split_id": split_id, "seed": cfg.SEED, "smoke_test": args.smoke, "created": datetime.now().strftime("%Y-%m-%d %H:%M")}
    (final_dir / "artifact.json").write_text(json.dumps(artifact, indent=2))
    pd.concat({n: c for n, c in curves.items()}, names=["class"]).reset_index(level=0).to_csv(final_dir / "thresholds_validation.csv", index=False)
    plot_threshold_curves(curves, thresholds, final_dir / "f1_vs_confidence.png")
    notes = ["3-CLASS MODEL: Crack, Pothole, Rut (the classes of the local labels). It is NOT a D00 / D10 / D20 / D40 model: the app and E4 code must use artifact.json 'classes'.",
             "Trained on the local TRAIN photos only, starting from model/final_model/best.pt (backbone and neck weights kept, class outputs new). Nothing from the RDD split is used here.",
             "Thresholds were chosen on the local VALIDATION photos; the local TEST photos were used once, for these tables.",
             "'original (merged)' = model/final_model/best.pt with ITS OWN thresholds (chosen on the RDD validation split), its D00 / D10 / D20 boxes merged into Crack "
             "(duplicates removed by NMS) and D40 = Pothole, on the same local test photos. It has no Rut class, so its Rut recall is 0 by design: "
             "compare with 'macro (Crack + Pothole)' for a fair look.",
             "The local sets are small (see the box counts): a class with few boxes in validation / test gives an unreliable threshold / precision / recall.",
             "Same settings as model/final_model.py (imported from it): model size, image size, epochs, batch size, seed, Ultralytics extras.",
             "best.pt = epoch with the best validation score; last.pt = final epoch. artifact.json belongs to best.pt.",
             "For the next experiment use ONLY the 'test' photos of local_split.csv: the train and validation photos were seen during training."]
    tables = {"test fine-tuned vs original": comparison, "local test metrics (fine-tuned)": tuned_table,
              "local test box metrics (mAP)": pd.DataFrame([box], index=["fine-tuned"]), "local photos and boxes per split": box_counts}
    settings = {**{k: v for k, v in artifact.items() if k != "confidence_thresholds"}, "confidence_thresholds": thresholds}

    # ---- the plain-text results FIRST (they must never depend on the Excel step): README + one csv per table + the tables on screen
    (final_dir / "README.md").write_text(
        "# Fine-tuned final model (Crack, Pothole, Rut)\n\n`model/final_model/best.pt` trained further on the local photos (train split of `local_split.csv`), "
        f"{epochs} epochs, same settings as `model/final_model.py`, but with the 3 LOCAL classes: {', '.join(LOCAL_CLASSES)}.\n\n"
        f"Thresholds (local validation): {thresholds}\n\nLocal test mAP@0.5 = {box['mAP50']:.3f}, mAP@0.5:0.95 = {box['mAP50-95']:.3f}.\n"
        "Only the `test` photos of `local_split.csv` may be used to judge it.\n\n"
        "Tables as csv: `tables/`. Notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    tables_dir = final_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    for table_name, table in tables.items():
        table.to_csv(tables_dir / (re.sub(r"[^A-Za-z0-9]+", "_", table_name).strip("_").lower() + ".csv"))
    print("\nlocal photos and boxes per split:"); print(box_counts.to_string())
    print("\nconfidence thresholds:", thresholds)
    print("\nlocal test photos - fine-tuned model:"); print(tuned_table.round(3).to_string())
    print("\nlocal test photos - original final model (merged classes) vs fine-tuned:"); print(comparison.round(3).to_string())
    print("\nmAP on the local test photos:", {k: round(v, 3) for k, v in box.items()})

    # ---- Excel + txt (project helper); if it fails the run is NOT lost: everything above is already saved
    try:
        save_results(final_dir, "Fine-tuned final model - local photos (Crack, Pothole, Rut)", tables, settings, notes)
    except Exception:
        import traceback
        traceback.print_exc()
        print("\nWARNING: results.xlsx / results.txt could not be written completely (error above). The tables are saved as csv in", tables_dir)
    print("\nFine-tuned model saved in", final_dir)


if __name__ == "__main__":
    main()
