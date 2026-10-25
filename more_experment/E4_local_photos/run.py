"""E4 (fine-tuned model) - test the FINE-TUNED model (fine-tuning/final_model: Crack, Pothole, Rut) on photos it has NEVER seen.
No training, no tuning: the model and the confidence thresholds are only APPLIED (thresholds come from fine-tuning/final_model/artifact.json).

  RDD-test  : the held-out test photos, labelled with the LOCAL classes (0 Crack, 1 Pothole, 2 Rut), like the fine-tuning data
    python more_experment/E4_local_photos/run.py --local-dir /content/local_test --label-set local --name RDD-test --out-dir /content/e4ft_output_test

  RDD-Data  : photos labelled with the OLD RDD classes (0 D00, 1 D10, 2 D20, 3 D40): does the fine-tuned model still find them?
    python more_experment/E4_local_photos/run.py --local-dir /content/rdd_data --label-set rdd --name RDD-Data --out-dir /content/e4ft_output_data
    The old labels are merged into the model's classes: D00 + D10 + D20 -> Crack, D40 -> Pothole. The old labels have no Rut, so Rut is NOT judged
    here (Rut detections are only counted).

Folder of the photos (YOLO format; an EMPTY label file = no damage):
    <local-dir>/images/p001.jpg
    <local-dir>/labels/p001.txt      class_id x_center y_center width height (values 0-1)

What it does (same as the first E4, for the fine-tuned model)
  1. loads fine-tuning/final_model/best.pt + its thresholds; photos that were used for fine-tuning (train / validation) are LEFT OUT
  2. runs the model on every photo and draws the detections: photos WHERE damage was detected, photos with NO damage detected, photos where the model was WRONG
  3. precision / recall / F1 / AP per class, photo-level results
  4. the ORIGINAL final model (model/final_model/best.pt, its own thresholds, D00 / D10 / D20 merged into Crack, D40 = Pothole, NMS) on the SAME photos,
     and the change between the two (the original has no Rut class: its Rut recall is 0 by design)
  5. (--label-set rdd) the original model with its own 4 classes, as in the first E4

PRIVACY: the pictures go to --out-dir (NOT into the repository, which is public).
Only numbers (xlsx / txt / csv / charts without photos) go to more_experment/E4_local_photos/<name>/result/.
"""
import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
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
FINE_TUNED_DIR = PROJECT_ROOT / "fine-tuning" / "final_model"
FINE_TUNING_FILE = PROJECT_ROOT / "fine-tuning" / "final_model.py"
ORIGINAL_DIR = PROJECT_ROOT / "model" / "final_model"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
COLORS = {"Crack": "#ff9100", "Pothole": "#ff1744", "Rut": "#00b8d4"}
COLOR_NAMES = {"Crack": "orange", "Pothole": "red", "Rut": "cyan"}


def load_module(name, path):
    if not Path(path).exists():
        sys.exit(f"{path} not found.")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ft = load_module("finetune_file", FINE_TUNING_FILE)          # LOCAL_CLASSES, ORIGINAL_TO_LOCAL, nms_per_class, source_key (single source of truth)
fm = ft.get_final()                                          # model/final_model.py (predict_boxes, match_predictions, ...)
LOCAL = ft.LOCAL_CLASSES


# ============================================================================== the photos and labels
def find_photos(folder):
    images = Path(folder) / "images"
    if not images.is_dir():
        sys.exit(f"{images} not found. Put your photos in <local-dir>/images/")
    photos = sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        sys.exit(f"no photos (.jpg/.png) in {images}")
    return photos


def check_labels(folder, photos, label_names):
    """Every photo needs a label file; ids must fit the label set. Returns {photo: label file}."""
    label_dir = Path(folder) / "labels"
    if not label_dir.is_dir():
        sys.exit(f"{label_dir} not found. E4 needs the labels.")
    problems, files = [], {}
    for photo in photos:
        label_file = label_dir / (photo.stem + ".txt")
        if not label_file.exists():
            problems.append(f"no label file for {photo.name} (create an EMPTY {photo.stem}.txt if the photo shows no damage)")
            continue
        for number, line in enumerate(label_file.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parts = line.split()
                assert len(parts) == 5, "need exactly 5 numbers"
                class_id, values = int(parts[0]), [float(v) for v in parts[1:5]]
                assert class_id in range(len(label_names)), f"class id must be 0-{len(label_names) - 1} ({', '.join(label_names)})"
                assert all(0 <= v <= 1 for v in values) and values[2] > 0 and values[3] > 0, "values must be 0-1 and width / height > 0"
            except (ValueError, AssertionError, IndexError) as error:
                problems.append(f"bad line {number} in {label_file.name}: '{line[:60]}' ({error})")
        files[photo] = label_file
    if problems:
        sys.exit("Problems with the labels:\n  - " + "\n  - ".join(problems[:15]) + ("\n  ... and more" if len(problems) > 15 else ""))
    return files


def drop_fine_tuning_photos(photos):
    """Photos that were used for fine-tuning (train / validation, by file name or source name) must not be tested. Returns (kept, dropped names)."""
    split_file = FINE_TUNED_DIR / "local_split.csv"
    if not split_file.exists():
        print(f"WARNING: {split_file} not found: cannot check whether the photos were used for fine-tuning.")
        return photos, []
    split = pd.read_csv(split_file)
    seen = split[split.split.isin(["train", "val"])]
    names, sources = set(seen.name), {ft.source_key(n) for n in seen.name}
    kept, dropped = [], []
    for photo in photos:
        (dropped if photo.name in names or ft.source_key(photo.name) in sources else kept).append(photo)
    return kept, [p.name for p in dropped]


# ============================================================================== predictions, merging, matching
def keep_above(pred, thresholds, names):
    """pred (N,6): x1,y1,x2,y2,conf,class -> only the boxes whose confidence reaches the threshold of their class."""
    if len(pred) == 0:
        return pred
    need = np.array([thresholds[names[int(c)]] for c in pred[:, 5]])
    return pred[pred[:, 4] >= need]


def merge_original(boxes, thresholds=None):
    """Original model boxes -> local classes: (optional own thresholds), D00 / D10 / D20 -> Crack, D40 -> Pothole, NMS per class."""
    if len(boxes) == 0:
        return np.zeros((0, 6))
    boxes = boxes.copy()
    names = [cfg.DAMAGE_CLASSES[int(c)] for c in boxes[:, 5]]
    if thresholds is not None:
        keep = np.array([b[4] >= thresholds[n] for b, n in zip(boxes, names)])
        boxes, names = boxes[keep], [n for n, k in zip(names, keep) if k]
    if len(boxes) == 0:
        return np.zeros((0, 6))
    boxes[:, 5] = [LOCAL.index(ft.ORIGINAL_TO_LOCAL[n]) for n in names]
    return ft.nms_per_class(fm, boxes)


def map_truth(truth, label_set):
    """Real boxes (class, x1, y1, x2, y2) -> local class ids. rdd: D00 / D10 / D20 -> Crack (0), D40 -> Pothole (1)."""
    if label_set == "local" or len(truth) == 0:
        return truth
    truth = truth.copy()
    truth[:, 0] = [LOCAL.index(ft.ORIGINAL_TO_LOCAL[cfg.DAMAGE_CLASSES[int(c)]]) for c in truth[:, 0]]
    return truth


def average_precision(confidences, is_correct, n_truth):
    """AP (all-point interpolation) for one class from matched predictions; NaN if there is no real box."""
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


def class_table(matches_threshold, matches_all, thresholds, classes):
    """Per class: precision / recall / F1 at the thresholds (matches_threshold: boxes that passed the thresholds already, or all boxes + thresholds),
    AP50 without threshold (matches_all). Plus macro rows."""
    rows = {}
    for c, name in enumerate(LOCAL):
        if name not in classes:
            continue
        conf, ok, n = matches_threshold[c]
        r = fm.curve_from_matches(conf, ok, n, [thresholds[name]], 1.0).iloc[0]
        rows[name] = {"threshold": thresholds[name], "precision": r.precision, "recall": r.recall,
                      "F1": 2 * r.precision * r.recall / (r.precision + r.recall) if r.precision + r.recall else 0.0,
                      "TP": int(r.tp), "FP": int(r.fp), "FN": int(r.fn), "real_boxes": n,
                      "AP50 (no threshold)": average_precision(*matches_all[c])}
    table = pd.DataFrame(rows).T
    macro_sets = ([("macro (all classes)", list(classes)), ("macro (Crack + Pothole)", ["Crack", "Pothole"])] if "Rut" in classes
                  else [("macro (Crack + Pothole)", list(classes))])
    for label, subset in macro_sets:
        table.loc[label] = {**table.loc[subset, ["precision", "recall", "F1", "AP50 (no threshold)"]].astype(float).mean().to_dict(),
                            "threshold": np.nan, "TP": table.loc[subset, "TP"].sum(), "FP": table.loc[subset, "FP"].sum(),
                            "FN": table.loc[subset, "FN"].sum(), "real_boxes": table.loc[subset, "real_boxes"].sum()}
    return table


def compare(original, tuned):
    out = pd.concat({"original (merged)": original[["precision", "recall", "F1", "AP50 (no threshold)"]].astype(float),
                     "fine-tuned": tuned[["precision", "recall", "F1", "AP50 (no threshold)"]].astype(float)}, axis=1)
    out.columns = [f"{model} {metric}" for model, metric in out.columns]
    for metric in ["precision", "recall", "F1", "AP50 (no threshold)"]:
        out[f"{metric} change"] = out[f"fine-tuned {metric}"] - out[f"original (merged) {metric}"]
    return out


def describe(detections):
    return ", ".join(f"{LOCAL[int(d[5])]} {d[4]:.2f}" for d in detections) or "nothing"


def photo_status(detections, truth, labelled_classes):
    """Short verdict per photo: all found / partly / MISSED / FALSE ALARM / correct: no damage. Only classes in labelled_classes are judged."""
    ids = [LOCAL.index(c) for c in labelled_classes]
    detections = detections[np.isin(detections[:, 5], ids)] if len(detections) else detections
    truth = truth[np.isin(truth[:, 0], ids)] if len(truth) else truth
    matches = fm.match_predictions([detections], [truth], n_classes=len(LOCAL))
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
    return status, tp, fp, fn, len(truth)


# ============================================================================== pictures
def open_photo(path, max_side=None):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    return image


def draw_photo(ax, image, detections, truth, title):
    ax.imshow(image); ax.axis("off"); ax.set_title(title, fontsize=9)
    width, height = image.size
    for c, x1, y1, x2, y2 in truth:                                         # your label: white dashed line on a black outline
        for colour, line_width, style in (("black", 4, "-"), ("white", 2, "--")):
            ax.add_patch(patches.Rectangle((x1 * width, y1 * height), (x2 - x1) * width, (y2 - y1) * height,
                                           fill=False, edgecolor=colour, linewidth=line_width, linestyle=style))
    for x1, y1, x2, y2, conf, c in detections:
        name = LOCAL[int(c)]
        ax.add_patch(patches.Rectangle((x1 * width, y1 * height), (x2 - x1) * width, (y2 - y1) * height, fill=False, edgecolor=COLORS[name], linewidth=3))
        ax.text(x1 * width, max(y1 * height - 4, 12), f"{name} {conf:.2f}", color="white", fontsize=9, weight="bold",
                bbox=dict(facecolor=COLORS[name], alpha=0.9, pad=1.5, edgecolor="none"))


def save_gallery(items, out_dir, prefix, headline, cols=3, rows=3):
    """items: list of (photo path, detections, truth, status). One PNG per page of cols*rows photos."""
    files = []
    per_page = cols * rows
    for page_number, start in enumerate(range(0, len(items), per_page), start=1):
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, (photo, detections, truth, status) in zip(axes, items[start:start + per_page]):
            draw_photo(ax, open_photo(photo, max_side=900), detections, truth, f"{photo.name[:40]}\n{status} | model: {describe(detections)}")
        fig.suptitle(f"{headline}   (page {page_number})   colours: " + ", ".join(f"{n}={COLOR_NAMES[n]}" for n in LOCAL) + "   white dashed = your label", fontsize=12)
        plt.tight_layout()
        path = Path(out_dir) / f"{prefix}_{page_number:02d}.png"
        fig.savefig(path, dpi=75, bbox_inches="tight"); plt.close(fig)
        files.append(path.name)
    return files


def save_single(photo, detections, truth, status, path):
    image = open_photo(photo, max_side=1600)
    fig, ax = plt.subplots(figsize=(10, 10 * image.size[1] / image.size[0] + 0.6))
    draw_photo(ax, image, detections, truth, f"{photo.name[:50]} - {status}\nmodel: {describe(detections)}")
    fig.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)


# ============================================================================== the experiment
def main():
    parser = argparse.ArgumentParser(description=NAME + " (fine-tuned model)")
    parser.add_argument("--local-dir", required=True, help="folder with images/ and labels/")
    parser.add_argument("--label-set", choices=["local", "rdd"], default="local",
                        help="local = labels 0 Crack, 1 Pothole, 2 Rut | rdd = old labels 0 D00, 1 D10, 2 D20, 3 D40 (merged into Crack / Pothole)")
    parser.add_argument("--name", required=True, help="result folder name: more_experment/E4_local_photos/<name>/result (e.g. RDD-test, RDD-Data)")
    parser.add_argument("--out-dir", default="", help="where the pictures go (default: <local-dir>/e4_finetuned_output)")
    parser.add_argument("--keep-seen", action="store_true", help="do NOT leave out photos used for fine-tuning (train / validation): results then say nothing about new photos")
    parser.add_argument("--allow-photos-in-repo", action="store_true", help="allow --out-dir inside the repository (it is PUBLIC)")
    args = parser.parse_args()

    folder = Path(args.local_dir)
    out_dir = Path(args.out_dir) if args.out_dir else folder / "e4_finetuned_output"
    if PROJECT_ROOT in out_dir.resolve().parents and not args.allow_photos_in_repo:
        sys.exit(f"{out_dir} is inside the repository, which is public. Choose a folder outside it for --out-dir.")
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
    result_dir = HERE / args.name / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    # ---- the fine-tuned model and the original model
    artifact_file, weights = FINE_TUNED_DIR / "artifact.json", FINE_TUNED_DIR / "best.pt"
    if not artifact_file.exists() or not weights.exists():
        sys.exit("fine-tuning/final_model/best.pt or artifact.json not found. Fine-tune first (python fine-tuning/final_model.py ...) and pull it from GitHub.")
    artifact = json.loads(artifact_file.read_text())
    if artifact.get("classes") != LOCAL:
        sys.exit(f"artifact.json has classes {artifact.get('classes')} but this script expects {LOCAL}.")
    thresholds, imgsz = artifact["confidence_thresholds"], artifact["image_size"]
    original_artifact = json.loads((ORIGINAL_DIR / "artifact.json").read_text())
    original_thresholds, original_imgsz = original_artifact["confidence_thresholds"], original_artifact["image_size"]

    # ---- the photos
    label_names = LOCAL if args.label_set == "local" else cfg.DAMAGE_CLASSES
    photos = find_photos(folder)
    if not args.keep_seen:
        photos, dropped = drop_fine_tuning_photos(photos)
        if dropped:
            print(f"left out {len(dropped)} photos that were used for fine-tuning (train / validation), e.g. {dropped[:3]}")
        if not photos:
            sys.exit("no photos left: all of them were used for fine-tuning. Use photos the model has never seen (or --keep-seen).")
    label_files = check_labels(folder, photos, label_names)
    judged = list(LOCAL) if args.label_set == "local" else ["Crack", "Pothole"]
    print(f"== {NAME} (fine-tuned) | {args.name} | label set: {args.label_set} | {len(photos)} photos | classes judged: {judged}")
    print("fine-tuned thresholds:", thresholds)

    # ---- run both models once; keep ALL boxes (for AP) and the thresholded ones (for the pictures)
    paths = [str(p) for p in photos]
    truths = [map_truth(fm.load_ground_truth(label_files[p]), args.label_set) for p in photos]
    tuned_all = fm.predict_boxes(weights, paths, imgsz)
    original_all = fm.predict_boxes(ORIGINAL_DIR / "best.pt", paths, original_imgsz)
    ignored_rut = 0
    if args.label_set == "rdd":                                              # the old labels have no Rut: Rut boxes are not judged
        rut = LOCAL.index("Rut")
        ignored_rut = int(sum(((b[:, 5] == rut) & (b[:, 4] >= thresholds["Rut"])).sum() for b in tuned_all if len(b)))
        tuned_all = [b[b[:, 5] != rut] if len(b) else b for b in tuned_all]
    tuned_detections = [keep_above(b, thresholds, LOCAL) for b in tuned_all]

    tuned_matches_all = fm.match_predictions(tuned_all, truths, n_classes=len(LOCAL))
    tuned = class_table(tuned_matches_all, tuned_matches_all, thresholds, judged)
    merged_thr = [merge_original(b, original_thresholds) for b in original_all]
    merged_all = [merge_original(b, None) for b in original_all]
    original = class_table(fm.match_predictions(merged_thr, truths, n_classes=len(LOCAL)),
                           fm.match_predictions(merged_all, truths, n_classes=len(LOCAL)), {c: 0.0 for c in LOCAL}, judged)
    comparison = compare(original, tuned)

    # ---- per photo + galleries
    items, rows = [], []
    for photo, det, truth in zip(photos, tuned_detections, truths):
        status, tp, fp, fn, n_real = photo_status(det, truth, judged)
        items.append((photo, det, truth, status))
        rows.append({"photo": photo.name, "status": status, "detections": describe(det), "n_detections": len(det),
                     "real_boxes": n_real, "found": tp, "false_alarms": fp, "missed": fn})
        save_single(photo, det, truth, status, out_dir / "annotated" / f"{photo.stem[:80]}_annotated.jpg")
    per_photo = pd.DataFrame(rows)
    damage = [i for i in items if len(i[1])]
    none = [i for i in items if not len(i[1])]
    wrong = [i for i in items if i[3] not in ("all found", "correct: no damage")]
    files = {"damage detected": save_gallery(damage, out_dir, "gallery_damage_detected", "Photos WHERE the fine-tuned model detected damage"),
             "no damage detected": save_gallery(none, out_dir, "gallery_no_damage_detected", "Photos where the fine-tuned model detected NO damage"),
             "mistakes": save_gallery(wrong, out_dir, "gallery_mistakes", "Photos where the fine-tuned model was WRONG (missed / false alarm / partly)")}

    with_damage, without = per_photo[per_photo.real_boxes > 0], per_photo[per_photo.real_boxes == 0]
    summary = {"photos": len(photos), "photos with damage detected": len(damage), "photos with NO damage detected": len(none),
               "photos WITH damage (labels)": len(with_damage), "...of which at least one damage found": int((with_damage.found > 0).sum()),
               "photos WITHOUT damage (labels)": len(without), "...of which the model raised a false alarm": int((without.n_detections > 0).sum())}
    if args.label_set == "rdd":
        summary["Rut detections (not judged: old labels have no Rut)"] = ignored_rut
    tables = {"summary": pd.DataFrame({"value": summary}), "fine-tuned per class": tuned, "original merged per class": original,
              "fine-tuned vs original": comparison, "per photo": per_photo}
    if args.label_set == "rdd":                                              # the original model as in the first E4: its own 4 classes
        own_thr = [keep_above(b, original_thresholds, cfg.DAMAGE_CLASSES) for b in original_all]
        own_truths = [fm.load_ground_truth(label_files[p]) for p in photos]
        tables["original own 4 classes"] = fm.metrics_at_thresholds(fm.match_predictions(own_thr, own_truths), original_thresholds)

    # ---- show + save the numbers (csv first: they never depend on the Excel step)
    for name, table in tables.items():
        if name != "per photo":
            print(f"\n{name}:"); print(table.round(3).to_string())
    tables_dir = result_dir / "tables"; tables_dir.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(tables_dir / (re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() + ".csv"))
    notes = ["No training, no tuning on these photos: the fine-tuned model and its thresholds (chosen on the fine-tuning VALIDATION photos) were only APPLIED.",
             "Photos that were used for fine-tuning (train / validation, by file name or source name) are left out, unless --keep-seen was given.",
             f"Label set '{args.label_set}': " + ("labels are the local classes Crack / Pothole / Rut." if args.label_set == "local" else
              "old RDD labels merged into Crack (D00 + D10 + D20) and Pothole (D40); Rut is not judged because these labels have no Rut."),
             "'original (merged)' = model/final_model/best.pt with its own thresholds, D00 / D10 / D20 merged into Crack (NMS), D40 = Pothole. "
             "It has no Rut class: its Rut recall is 0 by design; compare 'macro (Crack + Pothole)'.",
             "Solid coloured boxes = fine-tuned detections above the thresholds; white dashed = your labels.",
             "Pictures are in " + str(out_dir) + " (NOT in the repository: it is public).",
             "Small test sets give noisy numbers; a class with few real boxes has an unreliable precision / recall."]
    settings = {"experiment": NAME, "set": args.name, "label set": args.label_set, "photos": len(photos), "model": "fine-tuning/final_model (Crack, Pothole, Rut)",
                "image size": imgsz, "thresholds": thresholds, "fine-tuned split id": artifact.get("split_id"), "pictures written to": str(out_dir)}
    try:
        save_results(result_dir, f"E4 (fine-tuned model) - {args.name}", tables, settings, notes)
    except Exception:
        import traceback
        traceback.print_exc()
        print("\nWARNING: results.xlsx / results.txt could not be written completely (error above). The tables are saved as csv in", tables_dir)
    try:
        save_bar_chart(comparison.loc[judged][["original (merged) precision", "fine-tuned precision", "original (merged) recall", "fine-tuned recall"]],
                       result_dir / "e4_original_vs_finetuned.png", f"{args.name}: original (merged classes) vs fine-tuned model", "value")
    except Exception as error:
        print("chart not written:", error)
    print("\nPictures:", out_dir)
    for kind, names in files.items():
        print(f"  {kind}: {len(names)} page(s) -> {names[0] if names else '-'}")
    print("Numbers:", result_dir)


if __name__ == "__main__":
    main()
