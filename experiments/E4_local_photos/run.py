"""E4 - Test the FINAL model on our own local photos (no retraining, no new split).

    python experiments/E4_local_photos/run.py --local-dir /content/drive/MyDrive/rdd_local

What it does
  1. loads the final model (model/final_model/best.pt) and its confidence thresholds (artifact.json) - nothing is trained or tuned
  2. runs it on every photo in <local-dir>/images
  3. shows you the photos with the detections drawn on them: photos WHERE damage was detected, photos with NO damage detected,
     and (if you have labels) the photos where the model was wrong
  4. if <local-dir>/labels exists: precision / recall / F1 / AP per damage type, photo-level results, and the gap to the RDD test numbers

Your photos (YOLO format, class ids 0-3 = D00, D10, D20, D40; blur faces and licence plates):
    <local-dir>/images/p001.jpg
    <local-dir>/labels/p001.txt     one line per box: class_id x_center y_center width height (values 0-1); an EMPTY file = no damage
The labels folder is optional: without it you still get all the pictures and the detection counts.

PRIVACY: the pictures with detections are written to <local-dir>/e4_output (on your Drive), NOT into the repository,
because the repository is public. Only numbers (xlsx / txt / charts without photos) go to experiments/E4_local_photos/results/.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src import config as cfg
from src.reporting import save_bar_chart, save_results

NAME = "E4_local_photos"
FINAL_MODEL_FILE = PROJECT_ROOT / "model" / "final_model.py"
FINAL_MODEL_DIR = PROJECT_ROOT / "model" / "final_model"
LOCAL_DIR_ENV = "RDD_LOCAL_DIR"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
COLORS = {"D00": "#00c853", "D10": "#00b8d4", "D20": "#ff9100", "D40": "#ff1744"}      # colour per damage type
COLOR_NAMES = {"D00": "green", "D10": "cyan", "D20": "orange", "D40": "red"}
_FINAL = None


# ============================================================================== loading things
def local_dir(cli_value=""):
    value = cli_value or os.environ.get(LOCAL_DIR_ENV, "")
    return Path(value) if value else None


def get_final():
    """The code of model/final_model.py (its box matching and prediction functions), loaded once."""
    global _FINAL
    if _FINAL is None:
        if not FINAL_MODEL_FILE.exists():
            sys.exit(f"{FINAL_MODEL_FILE} not found. Create the final model first (python model/final_model.py).")
        spec = importlib.util.spec_from_file_location("final_model_file", FINAL_MODEL_FILE)
        _FINAL = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_FINAL)
    return _FINAL


def load_artifact():
    artifact_file, weights = FINAL_MODEL_DIR / "artifact.json", FINAL_MODEL_DIR / "best.pt"
    if not artifact_file.exists() or not weights.exists():
        sys.exit("model/final_model/best.pt or artifact.json not found. Run python model/final_model.py first "
                 "(and in Colab run the 'Download the project' cell so the saved final model is downloaded).")
    return json.loads(artifact_file.read_text()), weights


def find_photos(folder):
    images = Path(folder) / "images"
    if not images.is_dir():
        sys.exit(f"{images} not found. Put your photos in <local-dir>/images/")
    photos = sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        sys.exit(f"no photos (.jpg/.png) in {images}")
    return photos


def read_labels(folder, photos, use_labels=True):
    """Returns a list of arrays (class, x1, y1, x2, y2; relative) - one per photo - or None if there are no labels."""
    label_dir = Path(folder) / "labels"
    if not use_labels or not label_dir.is_dir():
        return None
    final = get_final()
    problems, truths = [], []
    for photo in photos:
        label_file = label_dir / (photo.stem + ".txt")
        if not label_file.exists():
            problems.append(f"no label file for {photo.name} (create an EMPTY {photo.stem}.txt if the photo shows no damage)")
            truths.append(None)
            continue
        for number, line in enumerate(label_file.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parts = line.split()
                class_id, values = int(parts[0]), [float(v) for v in parts[1:5]]
                assert len(values) == 4 and class_id in range(len(cfg.DAMAGE_CLASSES)) and all(0 <= v <= 1 for v in values) and values[2] > 0 and values[3] > 0
            except (ValueError, AssertionError, IndexError):
                problems.append(f"bad line {number} in {label_file.name}: '{line}' (need: class_id x_center y_center width height; class 0-3; values 0-1)")
        truths.append(final.load_ground_truth(label_file))
    if problems:
        sys.exit("Problems with the labels:\n  - " + "\n  - ".join(problems[:15]) +
                 ("\n  ... and more" if len(problems) > 15 else "") + "\nFix them, or run with --no-labels to only look at the detections.")
    return truths


# ============================================================================== metrics
def average_precision(confidences, is_correct, n_truth):
    """AP (all-point interpolation) for one damage type from matched predictions; NaN if there is no real box."""
    if n_truth == 0:
        return float("nan")
    if len(confidences) == 0:
        return 0.0
    order = np.argsort(-confidences)
    tp = np.cumsum(is_correct[order]); fp = np.cumsum(~is_correct[order])
    recall, precision = tp / n_truth, tp / (tp + fp)
    recall = np.concatenate([[0], recall, [1]]); precision = np.concatenate([[1], precision, [0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def keep_above_thresholds(prediction, thresholds):
    """prediction (N,6): x1,y1,x2,y2,conf,class -> only the boxes whose confidence reaches the threshold of their damage type."""
    if len(prediction) == 0:
        return prediction
    needed = np.array([thresholds[cfg.DAMAGE_CLASSES[int(c)]] for c in prediction[:, 5]])
    return prediction[prediction[:, 4] >= needed]


def describe(detections):
    return ", ".join(f"{cfg.DAMAGE_CLASSES[int(d[5])]} {d[4]:.2f}" for d in detections) or "nothing"


def photo_status(detections, truth, final):
    """Short verdict per photo. With labels: found / missed / false alarm / correct no-damage."""
    if truth is None:
        return ("damage detected" if len(detections) else "no damage detected"), 0, 0, 0
    matches = final.match_predictions([detections], [truth])
    tp = sum(int(m[1].sum()) for m in matches.values()); fp = sum(len(m[1]) - int(m[1].sum()) for m in matches.values())
    fn = len(truth) - tp
    if len(truth) == 0:
        status = "correct: no damage" if len(detections) == 0 else "FALSE ALARM"
    elif len(detections) == 0:
        status = "MISSED"
    elif fp == 0 and fn == 0:
        status = "all found"
    else:
        status = f"partly (found {tp}, missed {fn}, false {fp})"
    return status, tp, fp, fn


# ============================================================================== pictures
def open_photo(path, max_side=None):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")             # phone photos: apply the rotation stored in the file
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    return image


def draw_photo(ax, image, detections, truth, title):
    ax.imshow(image); ax.axis("off"); ax.set_title(title, fontsize=9)
    width, height = image.size
    if truth is not None:
        for c, x1, y1, x2, y2 in truth:                                         # your label: white dashed line on a black outline (visible on any road)
            for colour, line_width, style in (("black", 4, "-"), ("white", 2, "--")):
                ax.add_patch(patches.Rectangle((x1 * width, y1 * height), (x2 - x1) * width, (y2 - y1) * height,
                                               fill=False, edgecolor=colour, linewidth=line_width, linestyle=style))
    for x1, y1, x2, y2, conf, c in detections:
        name = cfg.DAMAGE_CLASSES[int(c)]
        ax.add_patch(patches.Rectangle((x1 * width, y1 * height), (x2 - x1) * width, (y2 - y1) * height,
                                       fill=False, edgecolor=COLORS[name], linewidth=3))
        ax.text(x1 * width, max(y1 * height - 4, 12), f"{name} {conf:.2f}", color="white", fontsize=9, weight="bold",
                bbox=dict(facecolor=COLORS[name], alpha=0.9, pad=1.5, edgecolor="none"))


def save_gallery(items, out_dir, prefix, headline, cols=3, rows=3):
    """items: list of (photo path, detections, truth, status). Writes one PNG per page of cols*rows photos. Returns the file names."""
    files = []
    per_page = cols * rows
    for page_number, start in enumerate(range(0, len(items), per_page), start=1):
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, (photo, detections, truth, status) in zip(axes, items[start:start + per_page]):
            draw_photo(ax, open_photo(photo, max_side=900), detections, truth, f"{photo.name}\n{status} | model: {describe(detections)}")
        fig.suptitle(f"{headline}   (page {page_number})   colours: " + ", ".join(f"{n}={COLOR_NAMES[n]}" for n in cfg.DAMAGE_CLASSES) +
                     ("   white dashed = your label" if any(t is not None for _, _, t, _ in items) else ""), fontsize=12)
        plt.tight_layout()
        path = Path(out_dir) / f"{prefix}_{page_number:02d}.png"
        fig.savefig(path, dpi=75, bbox_inches="tight"); plt.close(fig)
        files.append(path.name)
    return files


def save_single(photo, detections, truth, status, path):
    image = open_photo(photo, max_side=1600)
    fig, ax = plt.subplots(figsize=(10, 10 * image.size[1] / image.size[0] + 0.6))
    draw_photo(ax, image, detections, truth, f"{photo.name} - {status}\nmodel: {describe(detections)}")
    fig.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)


# ============================================================================== the experiment
def main():
    parser = argparse.ArgumentParser(description=NAME)
    parser.add_argument("--local-dir", default="", help=f"folder with images/ (and labels/); or set {LOCAL_DIR_ENV}")
    parser.add_argument("--out-dir", default="", help="where the pictures go (default: <local-dir>/e4_output)")
    parser.add_argument("--no-labels", action="store_true", help="ignore the labels folder: only look at the detections")
    parser.add_argument("--allow-photos-in-repo", action="store_true", help="allow --out-dir inside the repository (it is PUBLIC: only for photos you may publish)")
    args = parser.parse_args()

    folder = local_dir(args.local_dir)
    if folder is None:
        sys.exit(f"Give the folder with --local-dir or set the environment variable {LOCAL_DIR_ENV}")
    out_dir = Path(args.out_dir) if args.out_dir else folder / "e4_output"
    if PROJECT_ROOT in out_dir.resolve().parents and not args.allow_photos_in_repo:
        sys.exit(f"{out_dir} is inside the repository, which is public. Choose a folder on Google Drive for --out-dir.")
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
    results_dir = cfg.EXPERIMENTS_DIR / NAME / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    final = get_final()
    artifact, weights = load_artifact()
    thresholds, imgsz = artifact["confidence_thresholds"], artifact["image_size"]
    photos = find_photos(folder)
    truths = read_labels(folder, photos, use_labels=not args.no_labels)
    labelled = truths is not None
    print(f"== {NAME} | final model {artifact.get('model')} {imgsz}px | {len(photos)} photos | labels: {'yes' if labelled else 'no (pictures only)'}")
    print("confidence thresholds:", thresholds)

    # ---- run the final model once; keep ALL boxes (low confidence) for AP, and the thresholded boxes for the pictures
    all_boxes = final.predict_boxes(weights, [str(p) for p in photos], imgsz)
    detections = [keep_above_thresholds(b, thresholds) for b in all_boxes]
    truth_list = truths if labelled else [None] * len(photos)

    items, rows = [], []
    for photo, det, truth in zip(photos, detections, truth_list):
        status, tp, fp, fn = photo_status(det, truth, final)
        items.append((photo, det, truth, status))
        rows.append({"photo": photo.name, "status": status, "detections": describe(det), "n_detections": len(det),
                     "real_boxes": len(truth) if labelled else np.nan, "found": tp if labelled else np.nan,
                     "false_alarms": fp if labelled else np.nan, "missed": fn if labelled else np.nan})
        save_single(photo, det, truth, status, out_dir / "annotated" / f"{photo.stem}_annotated.jpg")
    per_photo = pd.DataFrame(rows)

    # ---- the galleries: damage detected / no damage detected / (with labels) mistakes
    damage = [i for i in items if len(i[1])]
    none = [i for i in items if not len(i[1])]
    files = {"damage detected": save_gallery(damage, out_dir, "gallery_damage_detected", "Photos WHERE the model detected damage"),
             "no damage detected": save_gallery(none, out_dir, "gallery_no_damage_detected", "Photos where the model detected NO damage")}
    if labelled:
        wrong = [i for i in items if i[3] not in ("all found", "correct: no damage")]
        files["mistakes"] = save_gallery(wrong, out_dir, "gallery_mistakes", "Photos where the model was WRONG (missed / false alarm / partly)")

    # ---- tables
    tables = {"per photo": per_photo}
    class_counts = pd.DataFrame({name: {"detections": int(sum((d[:, 5] == k).sum() for d in detections))}
                                 for k, name in enumerate(cfg.DAMAGE_CLASSES)}).T
    summary = {"photos": len(photos), "photos with damage detected": len(damage), "photos with NO damage detected": len(none)}
    settings_more = {}
    if labelled:
        matches_thr = final.match_predictions(detections, truth_list)                # boxes above the final thresholds
        matches_all = final.match_predictions(all_boxes, truth_list)                 # every box (for AP, no threshold)
        local = final.metrics_at_thresholds(matches_thr, thresholds)
        local["AP50 (no threshold)"] = [average_precision(*[matches_all[k][0], matches_all[k][1], matches_all[k][2]]) for k in range(4)] + [np.nan]
        local.loc["macro", "AP50 (no threshold)"] = np.nanmean(local["AP50 (no threshold)"].iloc[:4].astype(float))
        tables["local photos per damage type"] = local
        with_damage = per_photo[per_photo.real_boxes > 0]; without = per_photo[per_photo.real_boxes == 0]
        summary.update({"photos WITH damage (labels)": len(with_damage),
                        "...of which at least one damage found": int((with_damage.found > 0).sum()) if len(with_damage) else 0,
                        "photos WITHOUT damage (labels)": len(without),
                        "...of which the model raised a false alarm": int((without.n_detections > 0).sum()) if len(without) else 0})
        # gap to the RDD test numbers saved by model/final_model.py
        results_file = FINAL_MODEL_DIR / "results.xlsx"
        if results_file.exists():
            sheet = next((s for s in pd.ExcelFile(results_file).sheet_names if s.startswith("test metrics")), None)
            if sheet:
                rdd = pd.read_excel(results_file, sheet_name=sheet, index_col=0)
                gap = pd.DataFrame({f"{m} (local)": local[m].astype(float) for m in ["precision", "recall", "F1"]})
                for m in ["precision", "recall", "F1"]:
                    gap[f"{m} (RDD test)"] = rdd[m].astype(float); gap[f"{m} gap (RDD - local)"] = gap[f"{m} (RDD test)"] - gap[f"{m} (local)"]
                tables["gap RDD test vs local photos"] = gap
                save_bar_chart(gap[["precision (RDD test)", "precision (local)", "recall (RDD test)", "recall (local)"]].iloc[:4],
                               results_dir / "e4_rdd_vs_local.png", "Final model: RDD test vs our local photos (at the same thresholds)", "value")
        settings_more["real boxes in local photos"] = int(local["real_boxes"].iloc[:4].astype(float).sum())
    tables["summary"] = pd.DataFrame({"value": summary})
    tables["detections per damage type"] = class_counts

    notes = ["No training, no tuning on local photos: the final model and its thresholds (chosen on the RDD validation split) were only APPLIED.",
             "Solid coloured boxes = model detections above the final thresholds; white dashed = your labels.",
             "Pictures with the detections are in " + str(out_dir) + " (NOT in the repository: it is public).",
             "A small local set (40-80 photos) gives noisy numbers; a damage type with few real boxes has an unreliable precision/recall.",
             "Compare with the RDD test numbers in the sheet 'gap RDD test vs local photos' to describe the domain shift (camera, road, light, weather)."]
    xlsx, txt = save_results(results_dir, "E4 - Final model on our local photos", tables,
                             {"experiment": NAME, "final model": artifact.get("model"), "image size": imgsz, "thresholds": thresholds,
                              "photos": len(photos), "labels used": labelled, "pictures written to": str(out_dir),
                              "final model split id": artifact.get("split_id"), **settings_more}, notes)
    print(pd.DataFrame({"value": summary}).to_string())
    if labelled:
        print(tables["local photos per damage type"][["precision", "recall", "F1", "TP", "FP", "FN", "real_boxes", "AP50 (no threshold)"]].round(3))
    print("\nPictures:", out_dir)
    for kind, names in files.items():
        print(f"  {kind}: {len(names)} page(s) -> {names[0] if names else '-'}")
    print("Numbers:", xlsx, "and", txt)


if __name__ == "__main__":
    main()
