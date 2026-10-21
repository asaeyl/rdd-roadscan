"""E5 - Duplicate finder pilot: can an embedding find the matching damage crop again?  (no training, no cropping)

    python experiments/E5_Duplicate_finder/run.py --local-dir /content/rdd_test

Same experiment as E5_Duplicate_Finder_Pilot.ipynb. The zip is ALREADY prepared (crops, altered queries, pair labels):
    <dataset>/images/gallery/...            one cropped damage image per row of crop_labels.csv
    <dataset>/images/queries/query_0000.jpg lightly altered copy of gallery crop 0, query_0001.jpg ... (same order as crop_labels.csv)
    <dataset>/labels/crop_labels.csv        gallery_image, class_id, class_name, ...
    <dataset>/labels/pairs.csv              query_id, gallery_id, pair_label (1 same crop / 0 different), split (calibration / test)
<dataset> is --local-dir itself, or the folder inside it that holds images/ and labels/ (e.g. E5_Prepared_Dataset).

What it does
  1. loads the gallery crops, the query images and the labelled pairs
  2. compares three methods, all used as they are (no training): Perceptual Hash, ResNet50 (ImageNet), DINOv2 (ViT-S/14)
       - retrieval over the whole gallery: Recall@1, Recall@5, MRR
       - pair matching: the similarity threshold is chosen on the calibration pairs (best F1) and measured on the test pairs
         (precision, recall, F1, ROC-AUC)
  3. writes the numbers to experiments/E5_Duplicate_finder/results/ (no photos are written anywhere)

SCOPE: the positive pairs are synthetic views of the SAME crop, not two photos of the same real pothole.
The numbers measure matching under mild image changes, not real duplicate-report accuracy.
Needs internet on the first run (ResNet50 and DINOv2 weights are downloaded). A GPU is optional.
"""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from src import config as cfg
from src.reporting import save_bar_chart, save_results

NAME = "E5_Duplicate_finder"
LOCAL_DIR_ENV = "RDD_LOCAL_DIR"
ALL_METHODS = {"phash": "Perceptual Hash", "resnet50": "ResNet50 ImageNet", "dinov2": "DINOv2"}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ============================================================================== load the prepared dataset
def local_dir(cli_value=""):
    value = cli_value or os.environ.get(LOCAL_DIR_ENV, "")
    return Path(value) if value else None


def find_dataset(folder):
    """The folder that holds labels/crop_labels.csv: <folder> itself, or a folder inside it (E5_Prepared_Dataset, ...)."""
    folder = Path(folder)
    if (folder / "labels" / "crop_labels.csv").exists():
        return folder
    for found in sorted(folder.rglob("crop_labels.csv")):
        if found.parent.name == "labels":
            return found.parent.parent
    sys.exit(f"labels/crop_labels.csv not found in {folder}. The zip must contain images/gallery, images/queries, "
             "labels/crop_labels.csv and labels/pairs.csv (for example inside an E5_Prepared_Dataset folder).")


def load_rgb(path):
    if not Path(path).exists():
        sys.exit(f"image not found: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def load_dataset(data_dir):
    meta = pd.read_csv(data_dir / "labels" / "crop_labels.csv")
    pairs_file = data_dir / "labels" / "pairs.csv"
    if not pairs_file.exists():
        sys.exit(f"{pairs_file} not found.")
    pairs = pd.read_csv(pairs_file)
    missing = {"gallery_image", "class_id", "class_name"} - set(meta.columns)
    missing_pairs = {"query_id", "gallery_id", "pair_label", "split"} - set(pairs.columns)
    if missing or missing_pairs:
        sys.exit(f"missing columns: crop_labels.csv {sorted(missing)} | pairs.csv {sorted(missing_pairs)}")

    query_dir = data_dir / "images" / "queries"
    crops = [load_rgb(data_dir / p) for p in meta["gallery_image"]]
    queries = [load_rgb(query_dir / f"query_{i:04d}.jpg") for i in range(len(meta))]
    assert len(crops) == len(queries) == len(meta), "Image and crop-label counts do not match."
    assert set(pairs["split"]) == {"calibration", "test"}, "Expected calibration and test pair splits."
    assert {int(v) for v in pairs["pair_label"]} == {0, 1}, "pair_label must contain 0 and 1."
    assert pairs["query_id"].between(0, len(crops) - 1).all() and pairs["gallery_id"].between(0, len(crops) - 1).all(), "pair ids outside the gallery."
    return crops, queries, meta, pairs


# ============================================================================== features
def phash_features(images):
    """8x8 low-frequency DCT bits (without the DC term) = 63 bits per image."""
    from scipy.fft import dctn
    out = []
    for image in images:
        a = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        d = dctn(a, norm="ortho")[:8, :8].flatten()[1:]
        out.append((d > np.median(d)).astype(np.float32))
    return np.stack(out)


def encode_torch(images, model, transform, device, batch_size=32):
    import torch
    import torch.nn.functional as F
    out = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.stack([transform(x.convert("RGB")) for x in images[start:start + batch_size]]).to(device)
            z = model(batch)
            if isinstance(z, dict):
                z = z.get("x_norm_clstoken", next(iter(z.values())))
            out.append(F.normalize(z.reshape(z.shape[0], -1).float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def extract_features(methods, queries, crops):
    """{method name: (query features, gallery features, is_binary)} for the chosen methods."""
    features = {}
    if "phash" in methods:
        features[ALL_METHODS["phash"]] = (phash_features(queries), phash_features(crops), True)
    if "resnet50" in methods or "dinov2" in methods:
        import torch
        from torchvision import transforms
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", device)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        if "resnet50" in methods:
            from torchvision.models import resnet50, ResNet50_Weights
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
            encoder = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
            features[ALL_METHODS["resnet50"]] = (encode_torch(queries, encoder, transform, device), encode_torch(crops, encoder, transform, device), False)
        if "dinov2" in methods:
            try:
                dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
                features[ALL_METHODS["dinov2"]] = (encode_torch(queries, dino, transform, device), encode_torch(crops, dino, transform, device), False)
            except Exception as error:                                   # no internet / hub problem: keep the other methods
                print(f"WARNING: DINOv2 could not be loaded ({type(error).__name__}: {error}). It is left out of the results.")
    if not features:
        sys.exit("no method could be run.")
    return features


# ============================================================================== evaluation
def similarity_matrix(qfeat, gfeat, binary):
    """Normalised Hamming similarity for hash bits, cosine similarity for embeddings. Shape: (queries, gallery)."""
    if binary:
        different = qfeat @ (1 - gfeat).T + (1 - qfeat) @ gfeat.T
        return 1.0 - different / qfeat.shape[1]
    qn = qfeat / (np.linalg.norm(qfeat, axis=1, keepdims=True) + 1e-12)
    gn = gfeat / (np.linalg.norm(gfeat, axis=1, keepdims=True) + 1e-12)
    return qn @ gn.T


def evaluate(qfeat, gfeat, binary, source_ids, labeled_pairs):
    from sklearn.metrics import precision_recall_curve, roc_auc_score
    scores = similarity_matrix(qfeat, gfeat, binary)
    ranks = np.argsort(-scores, axis=1)
    recall1 = float(np.mean(ranks[:, 0] == source_ids))
    recall5 = float(np.mean([sid in row[:5] for sid, row in zip(source_ids, ranks)]))
    mrr = float(np.mean([1.0 / (np.where(row == sid)[0][0] + 1) for sid, row in zip(source_ids, ranks)]))

    pairs = labeled_pairs.copy()
    pairs["similarity"] = [scores[int(q), int(g)] for q, g in zip(pairs.query_id, pairs.gallery_id)]
    calibration, test = pairs[pairs.split == "calibration"], pairs[pairs.split == "test"]
    precision_curve, recall_curve, thresholds = precision_recall_curve(calibration.pair_label, calibration.similarity)
    f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-12)
    threshold = float(thresholds[int(np.nanargmax(f1_curve[:-1]))])

    y_true, y_score = test.pair_label.to_numpy(dtype=np.uint8), test.similarity.to_numpy()
    y_pred = y_score >= threshold
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp + 1e-12); recall = tp / (tp + fn + 1e-12)
    metrics = {"Recall@1": recall1, "Recall@5": recall5, "MRR": mrr,
               "ROC-AUC (test pairs)": float(roc_auc_score(y_true, y_score)),
               "Threshold (calibration)": threshold, "Precision (test pairs)": precision, "Recall (test pairs)": recall,
               "F1 (test pairs)": 2 * precision * recall / (precision + recall + 1e-12)}
    return metrics, pairs


# ============================================================================== the experiment
def main():
    parser = argparse.ArgumentParser(description=NAME)
    parser.add_argument("--local-dir", default="", help=f"the unzipped dataset (images/ + labels/); or set {LOCAL_DIR_ENV}")
    parser.add_argument("--result-dir", default="", help=f"where the numbers go (default: experiments/{NAME}/results)")
    parser.add_argument("--methods", default="phash,resnet50,dinov2", help="comma list from: phash, resnet50, dinov2")
    args = parser.parse_args()

    folder = local_dir(args.local_dir)
    if folder is None:
        sys.exit(f"Give the folder with --local-dir or set the environment variable {LOCAL_DIR_ENV}")
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        sys.exit(f"unknown method(s) {unknown}; choose from {list(ALL_METHODS)}")
    results_dir = Path(args.result_dir) if args.result_dir else cfg.EXPERIMENTS_DIR / NAME / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_dir = find_dataset(folder)
    crops, queries, meta, pairs = load_dataset(data_dir)
    source_ids = np.arange(len(crops))
    print(f"== {NAME} | dataset {data_dir} | methods: {', '.join(ALL_METHODS[m] for m in methods)}")
    print(f"gallery crops: {len(crops)} | query images: {len(queries)} | labelled pairs: {len(pairs)} "
          f"(positive {(pairs.pair_label == 1).sum()}, negative {(pairs.pair_label == 0).sum()})")

    class_table = meta["class_name"].value_counts().rename("crops").to_frame()
    split_table = pairs.groupby(["split", "pair_label"]).size().unstack(fill_value=0)
    split_table.columns = [f"{'positive' if c == 1 else 'negative'} ({c})" for c in split_table.columns]
    print(class_table.T.to_string()); print(split_table.to_string())

    features = extract_features(methods, queries, crops)
    rows, pair_scores = [], pairs.copy()
    for method, (qfeat, gfeat, binary) in features.items():
        metrics, scored = evaluate(qfeat, gfeat, binary, source_ids, pairs)
        rows.append({"Method": method, **metrics})
        pair_scores[f"{method}_similarity"] = scored["similarity"].to_numpy()
    results = pd.DataFrame(rows).set_index("Method")
    print(results.round(4).to_string())

    results.to_csv(results_dir / "E5_results.csv")
    pair_scores.to_csv(results_dir / "E5_pair_scores.csv", index=False)
    save_bar_chart(results[["Recall@1", "Recall@5", "Precision (test pairs)", "Recall (test pairs)", "F1 (test pairs)", "ROC-AUC (test pairs)"]],
                   results_dir / "e5_methods.png", "E5: damage-crop retrieval and pair matching (synthetic views)", "score")
    notes = ["SYNTHETIC-VIEW PILOT: every positive pair is a lightly altered copy of the same crop, not a second photo of the same real damage. "
             "Do not read these numbers as accuracy on real duplicate reports.",
             "Nothing is trained and no image is cropped here: the crops, queries and pair labels come ready-made from the dataset zip.",
             "The threshold of each method is chosen on the calibration pairs (best F1) and measured on the test pairs; ROC-AUC uses the balanced test pairs.",
             "Retrieval (Recall@1, Recall@5, MRR) ranks the true crop against the WHOLE gallery. Road photos from the same drive are often near-identical, "
             "so a different crop can look like the right one: this lowers Recall@1 for every method."]
    xlsx, txt = save_results(results_dir, "E5 - Duplicate finder (embedding comparison)",
                             {"results": results, "crops per damage type": class_table, "pairs per split": split_table},
                             {"experiment": NAME, "dataset folder": str(data_dir), "gallery crops": len(crops), "query images": len(queries),
                              "labelled pairs": len(pairs), "methods": ", ".join(features)}, notes)
    print("\nNumbers:", results_dir, "|", xlsx, "and", txt)


if __name__ == "__main__":
    main()
