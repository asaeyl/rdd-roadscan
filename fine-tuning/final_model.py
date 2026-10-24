"""FINE-TUNED FINAL MODEL = the final model (model/final_model/best.pt) trained further on OUR local photos.

python fine-tuning/final_model.py --local-dir /content/rdd_local --check-only  # BEFORE training: do the labels mean the same damage classes as the model?
python fine-tuning/final_model.py --local-dir /content/rdd_local --smoke   # 1 epoch on 5% of the data: checks that everything works
python fine-tuning/final_model.py --local-dir /content/rdd_local           # the real run: 20 epochs, same settings as model/final_model.py
python fine-tuning/final_model.py --local-dir /content/rdd_local --reuse   # do NOT train again: redo thresholds + evaluation only

Everything except the data is taken FROM model/final_model.py (it is imported, not copied): model size, image size (640),
epochs (20), batch size, seed, Ultralytics settings, the box matching and the threshold choice. Only two things change:
  - the training starts from model/final_model/best.pt (not from COCO weights)
  - the training / validation / test photos are YOUR local photos (nothing from the RDD split is used)

Your photos (the same YOLO folder as E4, class ids 0-3 = D00, D10, D20, D40; an EMPTY label file = no damage):
    <local-dir>/images/p001.jpg
    <local-dir>/labels/p001.txt

The local photos (820 at the moment) are split ONCE into train / validation / test (default 70 / 15 / 15 = about 574 / 123 / 123). A photo in train is never used to judge the model.
Photos with neighbouring file names are often near-identical frames, so the split keeps blocks of consecutive file names together
(--block-size). The split is saved in <output>/local_split.csv: the next experiment (E4 on the fine-tuned model) must use ONLY the 'test' photos.

What it does
  1. checks the photos and labels, splits them, builds a YOLO folder in the work directory (not in the repository)
  2. fine-tunes best.pt on the local TRAIN photos (20 epochs, validation on the local VALIDATION photos)
  3. picks ONE confidence threshold per damage type on the local VALIDATION photos (same cost-aware F-beta as the final model)
  4. evaluates ONCE on the local TEST photos, next to the original final model (with its own thresholds) on the same photos
  5. saves everything in fine-tuning/final_model/ : best.pt, last.pt, artifact.json (same format: RoadDamageDetector can load it), results, plots
"""
import argparse
import hashlib
import importlib.util
import json
import os
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
_FINAL = None


# ============================================================================== loading things
def get_final():
    """model/final_model.py loaded as a module: all its settings and functions are re-used, so nothing can drift apart."""
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


# ============================================================================== the local photos
def check_local_data(folder):
    """Every photo needs a label file (EMPTY = no damage). Returns {photo name: (image path, label path)}."""
    images, labels = Path(folder) / "images", Path(folder) / "labels"
    if not images.is_dir() or not labels.is_dir():
        sys.exit(f"{images} and {labels} are needed (the same folders as in E4).")
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
                class_id, values = int(parts[0]), [float(v) for v in parts[1:5]]
                assert len(values) == 4 and class_id in range(len(cfg.DAMAGE_CLASSES)) and all(0 <= v <= 1 for v in values) and values[2] > 0 and values[3] > 0
            except (ValueError, AssertionError, IndexError):
                problems.append(f"bad line {number} in {label_file.name}: '{line}' (need: class_id x_center y_center width height; class 0-3; values 0-1)")
        found[photo.name] = (photo, label_file)
    if problems:
        sys.exit("Problems with the local photos:\n  - " + "\n  - ".join(problems[:15]) + ("\n  ... and more" if len(problems) > 15 else ""))
    return found


def make_split(names, val_share, test_share, block_size, seed):
    """train / val / test by BLOCKS of consecutive file names (near-identical neighbouring frames never end up in different splits)."""
    names = sorted(names)
    blocks = [names[i:i + block_size] for i in range(0, len(names), block_size)]
    order = np.random.default_rng(seed).permutation(len(blocks))
    target_test, target_val = round(len(names) * test_share), round(len(names) * val_share)
    assignment, counts = {}, {"test": 0, "val": 0}
    for index in order:
        if counts["test"] < target_test:
            split = "test"
        elif counts["val"] < target_val:
            split = "val"
        else:
            split = "train"
        for name in blocks[index]:
            assignment[name] = split
        if split in counts:
            counts[split] += len(blocks[index])
    table = pd.DataFrame({"name": names, "split": [assignment[n] for n in names]})
    if min((table.split == s).sum() for s in ("train", "val", "test")) == 0:
        sys.exit("A split came out empty: use more photos, a smaller --block-size or other shares.")
    return table


def count_boxes(split_table, photos):
    """Photos and boxes per split and damage type (so you can see if a split has no examples of a damage type)."""
    rows = {}
    for split in ("train", "val", "test"):
        names = split_table[split_table.split == split].name
        counts = {name: 0 for name in cfg.DAMAGE_CLASSES}
        for n in names:
            for line in photos[n][1].read_text().splitlines():
                if line.strip():
                    counts[cfg.DAMAGE_CLASSES[int(line.split()[0])]] += 1
        rows[split] = {"photos": len(names), **counts}
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
                                         "names": {i: n for i, n in enumerate(cfg.DAMAGE_CLASSES)}}, sort_keys=False))
    return data_yaml


def paths_for(split_table, split):
    """(image files, label files) of the local YOLO folder for one split."""
    root = cfg.WORK_DIR / "yolo_local"
    names = list(split_table[split_table.split == split].name)
    return ([str(root / "images" / split / n) for n in names], [root / "labels" / split / (Path(n).stem + ".txt") for n in names])


def split_fingerprint(split_table):
    return hashlib.sha1(split_table.sort_values("name").to_csv(index=False).encode()).hexdigest()[:10]


# ============================================================================== evaluation helpers
def evaluate_on(fm, weights, image_size, split_table, split):
    """Run a model on one local split and match its boxes to the real ones. Returns the matches."""
    images, labels = paths_for(split_table, split)
    matches = fm.match_predictions(fm.predict_boxes(weights, images, image_size), [fm.load_ground_truth(p) for p in labels])
    return matches


def compare_tables(original, tuned):
    columns = ["precision", "recall", "F1"]
    out = pd.concat({"original": original[columns].astype(float), "fine-tuned": tuned[columns].astype(float)}, axis=1)
    out.columns = [f"{model} {metric}" for model, metric in out.columns]
    for metric in columns:
        out[f"{metric} change"] = out[f"fine-tuned {metric}"] - out[f"original {metric}"]
    return out



# ============================================================================== do the labels mean the same as the model's classes?
def check_model_classes(weights):
    """The class names stored inside the weights must be exactly D00, D10, D20, D40 in this order (= the label ids 0-3)."""
    from ultralytics import YOLO
    names = {int(k): v for k, v in YOLO(str(weights)).names.items()}
    expected = {i: n for i, n in enumerate(cfg.DAMAGE_CLASSES)}
    print("class names inside the weights:", names)
    if names != expected:
        sys.exit(f"The weights use {names} but the project (and your label ids 0-3) use {expected}. Fix this before training.")
    return names


def check_labels(fm, weights, photos, out_dir, min_conf=0.25, samples_per_class=6):
    """Does label id k mean the same damage as the model's class k?
    1) per damage type: how many boxes, in how many photos
    2) the ORIGINAL model (confidence >= min_conf) is run on the photos; every labelled box is compared with the model box that overlaps it
       best (IoU >= 0.5, any class): a table 'label class' x 'class the model sees there'. A big off-diagonal number = labels and model disagree.
    3) a picture with cropped label boxes per damage type, to look at (written to out_dir, NOT into the repository)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    names = sorted(photos)
    predictions = fm.predict_boxes(weights, [str(photos[n][0]) for n in names], fm.IMAGE_SIZE)
    rows, boxes_by_class = [], {c: [] for c in cfg.DAMAGE_CLASSES}
    for name, pred in zip(names, predictions):
        pred = pred[pred[:, 4] >= min_conf]
        for row in fm.load_ground_truth(photos[name][1]):
            label = cfg.DAMAGE_CLASSES[int(row[0])]
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
    confusion = pd.crosstab(table["label"], table["model sees"]).reindex(index=cfg.DAMAGE_CLASSES, columns=columns, fill_value=0)
    summary = pd.DataFrame({"boxes": confusion.sum(axis=1),
                            "photos": [table[table.label == c].photo.nunique() for c in cfg.DAMAGE_CLASSES],
                            "model sees the SAME class": [confusion.loc[c, c] for c in cfg.DAMAGE_CLASSES],
                            "model sees ANOTHER class": [confusion.loc[c, cfg.DAMAGE_CLASSES].sum() - confusion.loc[c, c] for c in cfg.DAMAGE_CLASSES],
                            "model sees nothing": confusion["no model box"]})
    print("\nlabelled boxes per damage type and what the ORIGINAL model sees there (confidence >= %.2f):" % min_conf)
    print(summary.to_string()); print("\nrows = your label, columns = class of the model box that overlaps it:"); print(confusion.to_string())
    for c in cfg.DAMAGE_CLASSES:
        matched = confusion.loc[c, cfg.DAMAGE_CLASSES]
        if matched.sum() >= 5 and matched.idxmax() != c:
            print(f"WARNING: your label {c} is mostly seen by the model as {matched.idxmax()} ({int(matched.max())} of {int(matched.sum())} matched boxes). "
                  f"Check whether the class ids of your labels are in the same order as D00, D10, D20, D40.")
    # pictures to look at
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.SEED)
    fig, axes = plt.subplots(len(cfg.DAMAGE_CLASSES), samples_per_class, figsize=(2.6 * samples_per_class, 2.6 * len(cfg.DAMAGE_CLASSES)))
    for r, c in enumerate(cfg.DAMAGE_CLASSES):
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
    parser = argparse.ArgumentParser(description="Fine-tune the final model on the local photos (same settings as model/final_model.py)")
    parser.add_argument("--local-dir", default="", help=f"folder with images/ and labels/; or set {LOCAL_DIR_ENV}")
    parser.add_argument("--base-weights", default="", help="weights to start from (default: model/final_model/best.pt)")
    parser.add_argument("--check-only", action="store_true", help="only check the labels against the original model (no training), then stop")
    parser.add_argument("--out-dir", default="", help="where the label-check pictures go (default: <local-dir>/label_check_output)")
    parser.add_argument("--smoke", action="store_true", help="quick test: 1 epoch on 5%% of the data, output in fine-tuning/final_model_smoke/")
    parser.add_argument("--reuse", action="store_true", help="do not train: use the saved fine-tuning/final_model/best.pt, redo thresholds + evaluation")
    parser.add_argument("--epochs", type=int, default=fm.EPOCHS, help=f"default {fm.EPOCHS} (as in model/final_model.py)")
    parser.add_argument("--val-share", type=float, default=0.15, help="share of the local photos for validation (default 0.15)")
    parser.add_argument("--test-share", type=float, default=0.15, help="share of the local photos for the final test (default 0.15)")
    parser.add_argument("--block-size", type=int, default=10, help="consecutive file names that stay in the same split (default 10)")
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

    epochs = 1 if args.smoke else args.epochs
    final_dir = FINE_TUNED_DIR.parent / ("final_model_smoke" if args.smoke else "final_model")
    final_dir.mkdir(parents=True, exist_ok=True)
    set_seed()

    # ---- 0) the local photos: check, split, build the YOLO folder
    photos = check_local_data(folder)
    check_model_classes(base)                                              # stops if the weights do not use D00, D10, D20, D40 (ids 0-3)
    if args.check_only:
        out_dir = Path(args.out_dir) if args.out_dir else folder / "label_check_output"
        if PROJECT_ROOT in out_dir.resolve().parents:
            sys.exit(f"{out_dir} is inside the repository, which is public. Choose another folder for --out-dir.")
        check_labels(fm, ORIGINAL_DIR / "best.pt" if not args.base_weights else base, photos, out_dir)
        return
    split_file = final_dir / "local_split.csv"
    if args.reuse and split_file.exists():
        split_table = pd.read_csv(split_file)
        if set(split_table.name) != set(photos):
            sys.exit("The photos changed since the saved split (local_split.csv). Run without --reuse.")
    else:
        split_table = make_split(photos, args.val_share, args.test_share, args.block_size, cfg.SEED)
    split_table.to_csv(split_file, index=False)
    data_yaml = build_yolo_local(split_table, photos)
    split_id = split_fingerprint(split_table)
    box_counts = count_boxes(split_table, photos)
    print(f"== fine-tuning | start: {base} | {fm.BASE_WEIGHTS} | {fm.IMAGE_SIZE} px | {epochs} epochs | local photos {len(photos)} | split id {split_id}"
          f"{' | SMOKE TEST' if args.smoke else ''}")
    print(box_counts.to_string())
    for split in ("val", "test"):
        empty = [n for n in cfg.DAMAGE_CLASSES if box_counts.loc[split, n] == 0]
        if empty:
            print(f"WARNING: the local {split} split has no {', '.join(empty)} boxes: "
                  + ("their thresholds fall back to the default." if split == "val" else "their test numbers are not meaningful."))

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
    matches_val = evaluate_on(fm, best, fm.IMAGE_SIZE, split_table, "val")
    thresholds, curves = fm.choose_thresholds(matches_val)
    print("confidence thresholds (chosen on the local validation photos):", thresholds)

    # ---- 3) the local TEST photos, once: fine-tuned model and (for comparison) the original final model on the same photos
    tuned_table = fm.metrics_at_thresholds(evaluate_on(fm, best, fm.IMAGE_SIZE, split_table, "test"), thresholds)
    original_table = fm.metrics_at_thresholds(
        evaluate_on(fm, ORIGINAL_DIR / "best.pt", original_artifact["image_size"], split_table, "test"), original_artifact["confidence_thresholds"])
    comparison = compare_tables(original_table, tuned_table)
    box = fm.evaluate_with_plots(best, data_yaml, fm.IMAGE_SIZE, final_dir / "plots")           # mAP on the local test photos

    # ---- 4) write fine-tuning/final_model/
    artifact = {"model": fm.BASE_WEIGHTS, "source": f"fine-tuned from {base.as_posix()}: {fm.IMAGE_SIZE} px, {epochs} epochs, on local photos"
                if not args.reuse else "re-used saved fine-tuned weights",
                "fine_tuned_from": base.as_posix(),
                "image_size": fm.IMAGE_SIZE, "epochs": run_args.get("epochs", epochs), "optimizer": run_args.get("optimizer"),
                "classes": cfg.DAMAGE_CLASSES, "confidence_thresholds": thresholds, "beta_per_class": fm.sf.BETA_PER_CLASS,
                "thresholds_chosen_on": "local validation photos", "iou_threshold": fm.sf.IOU_THRESHOLD,
                "local_test_mAP50": box["mAP50"], "local_test_mAP50-95": box["mAP50-95"],
                "local_photos": {s: int(box_counts.loc[s, "photos"]) for s in ("train", "val", "test")},
                "split_id": split_id, "seed": cfg.SEED, "smoke_test": args.smoke, "created": datetime.now().strftime("%Y-%m-%d %H:%M")}
    (final_dir / "artifact.json").write_text(json.dumps(artifact, indent=2))
    pd.concat({n: c for n, c in curves.items()}, names=["class"]).reset_index(level=0).to_csv(final_dir / "thresholds_validation.csv", index=False)
    fm.plot_threshold_curves(curves, thresholds, final_dir / "f1_vs_confidence.png")
    notes = ["Trained on the local TRAIN photos only, starting from model/final_model/best.pt. Nothing from the RDD split is used here.",
             "Thresholds were chosen on the local VALIDATION photos; the local TEST photos were used once, for these tables.",
             "'original' = model/final_model/best.pt with ITS OWN thresholds (chosen on the RDD validation split), on the same local test photos.",
             "The local sets are small: a damage type with few boxes in validation / test gives an unreliable threshold / precision / recall (see the box counts).",
             "Same settings as model/final_model.py (imported from it): model size, image size, epochs, batch size, seed, Ultralytics extras.",
             "best.pt = epoch with the best validation score; last.pt = final epoch. artifact.json belongs to best.pt.",
             "For the next experiment use ONLY the 'test' photos of local_split.csv: the train and validation photos were seen during training."]
    save_results(final_dir, "Fine-tuned final model - local photos",
                 {"local test: fine-tuned vs original": comparison, "local test metrics (fine-tuned)": tuned_table,
                  "local test box metrics (mAP)": pd.DataFrame([box], index=["fine-tuned"]), "local photos and boxes per split": box_counts},
                 {**{k: v for k, v in artifact.items() if k != "confidence_thresholds"}, "confidence_thresholds": thresholds}, notes)
    (final_dir / "README.md").write_text(
        "# Fine-tuned final model\n\n`model/final_model/best.pt` trained further on the local photos (train split of `local_split.csv`), "
        f"{epochs} epochs, same settings as `model/final_model.py`.\n\n"
        "Load it like the final model: `RoadDamageDetector(folder='fine-tuning/final_model')` (defined in `model/final_model.py`).\n\n"
        f"Thresholds (local validation): {thresholds}\n\nLocal test mAP@0.5 = {box['mAP50']:.3f}, mAP@0.5:0.95 = {box['mAP50-95']:.3f}.\n"
        "Only the `test` photos of `local_split.csv` may be used to judge it.\n")
    print("\nlocal test photos - original final model vs fine-tuned:")
    print(comparison.round(3).to_string())
    print("\nFine-tuned model saved in", final_dir)


if __name__ == "__main__":
    main()
