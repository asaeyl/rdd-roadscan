"""scripts/explore_split.py

Explore the frozen split file (data/splits_*.csv) and save a full report.

Handles two CSV layouts:
  A) one row per image, single 'class' / 'label' column
  B) one row per image, per-class count columns (count_D00, ...) and/or
     binary 'has_D00' columns - this is what splits_12000.csv uses.

Outputs (all in data/split_explore/):
  summary.txt            overview.xlsx
  split_sizes.png        class_balance.png       per_split_classes.png
  boxes_per_class.png    boxes_hist.png          coverage.png
  sample_grid.png        (only if images can be located)

Usage:
    python scripts/explore_split.py
    python scripts/explore_split.py --images-dir data/raw --n 16
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

try:
    from src import config as cfg
    DATA_DIR = Path(cfg.DATA_DIR)
    DAMAGE_CLASSES = list(getattr(cfg, "DAMAGE_CLASSES", ["D00", "D10", "D20", "D40"]))
except Exception as e:
    print("Could not import src.config, using defaults:", e)
    DATA_DIR = Path("data")
    DAMAGE_CLASSES = ["D00", "D10", "D20", "D40"]

_csvs = sorted(DATA_DIR.glob("splits_*.csv"))
DEFAULT_CSV = _csvs[-1] if _csvs else DATA_DIR / "splits_12000.csv"
DEFAULT_OUT = DATA_DIR / "split_explore"


# ---------------------------------------------------------------- args

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--images-dir", default=None,
                    help="where the JPEGs live; if not given, common places are tried")
    ap.add_argument("--n", type=int, default=16, help="images in the sample grid")
    return ap.parse_args()


# ---------------------------------------------------------------- column detection

def detect_columns(df, damage_classes):
    info = {"split": None, "label": None, "path": None, "name": None,
            "count_cols": {}, "has_cols": {}}

    for c in df.columns:
        cl = c.lower()
        if cl in ("split", "subset", "set", "fold") and info["split"] is None:
            info["split"] = c
        elif cl in ("class", "label", "damage", "damage_class", "category") and info["label"] is None:
            info["label"] = c
        elif cl in ("path", "image", "image_path", "filename", "file") and info["path"] is None:
            info["path"] = c
        elif cl in ("name", "image_name") and info["name"] is None:
            info["name"] = c

    for cls in damage_classes:
        if f"count_{cls}" in df.columns:
            info["count_cols"][cls] = f"count_{cls}"
        if f"has_{cls}" in df.columns:
            info["has_cols"][cls] = f"has_{cls}"
    return info


# ---------------------------------------------------------------- tables

def build_tables(df, info, damage_classes):
    tables = {}

    if info["split"]:
        t = df[info["split"]].value_counts().sort_index().rename("images").to_frame()
        t.index.name = "split"
        tables["images_per_split"] = t

    if info["has_cols"]:
        rows = {cls: int(df[col].astype(bool).sum())
                for cls, col in info["has_cols"].items()}
        t = pd.Series(rows, name="images").to_frame()
        t.index.name = "class"
        tables["class_image_counts"] = t
        if info["split"]:
            cros = pd.DataFrame({cls: df.groupby(info["split"])[col].sum().astype(int)
                                 for cls, col in info["has_cols"].items()})
            cros.index.name = "split"
            tables["per_split_crosstab"] = cros
    elif info["label"]:
        t = df[info["label"]].value_counts().sort_index().rename("images").to_frame()
        t.index.name = "class"
        tables["class_image_counts"] = t
        if info["split"]:
            tables["per_split_crosstab"] = pd.crosstab(df[info["split"]], df[info["label"]])

    if info["count_cols"]:
        rows = {cls: int(df[col].sum()) for cls, col in info["count_cols"].items()}
        t = pd.Series(rows, name="boxes").to_frame()
        t.index.name = "class"
        tables["class_box_counts"] = t

    if "n_boxes" in df.columns and info["split"]:
        tables["n_boxes_stats"] = df.groupby(info["split"])["n_boxes"].describe()

    # coverage tables: original_split vs new split, and has_no_damage vs new split
    if info["split"] and "original_split" in df.columns:
        cros = pd.crosstab(df["original_split"], df[info["split"]])
        cros.index.name = "original_split"
        tables["original_vs_new_split"] = cros
    if info["split"] and "has_no_damage" in df.columns:
        cros = pd.crosstab(df[info["split"]], df["has_no_damage"])
        cros.index.name = "split"
        tables["has_no_damage_per_split"] = cros
    if info["split"] and "country" in df.columns:
        cros = pd.crosstab(df["country"], df[info["split"]])
        cros.index.name = "country"
        tables["country_per_split"] = cros

    return tables


# ---------------------------------------------------------------- figures

def _annotate(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)


def fig_split_sizes(t, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    t["images"].plot.bar(ax=ax, color="#4C72B0")
    ax.set_title("Images per split"); ax.set_ylabel("images"); ax.set_xlabel("")
    _annotate(ax, t["images"].values)
    plt.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_class_image_counts(t, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    t["images"].plot.bar(ax=ax, color="#55A868")
    ax.set_title("Images containing each class (all splits)")
    ax.set_ylabel("images"); ax.set_xlabel("")
    _annotate(ax, t["images"].values)
    plt.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_per_split(cros, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    cros.plot.bar(ax=ax)
    ax.set_title("Images per class, per split")
    ax.set_ylabel("images"); ax.set_xlabel("")
    ax.legend(title="class", fontsize=8)
    plt.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_box_counts(t, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    t["boxes"].plot.bar(ax=ax, color="#C44E52")
    ax.set_title("Total boxes per class (all splits)")
    ax.set_ylabel("boxes"); ax.set_xlabel("")
    _annotate(ax, t["boxes"].values)
    plt.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_boxes_hist(df, split_col, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    if split_col:
        for s, g in df.groupby(split_col):
            ax.hist(g["n_boxes"], bins=40, alpha=0.5, label=str(s))
        ax.legend(title="split")
    else:
        ax.hist(df["n_boxes"], bins=40)
    ax.set_title("Boxes per image"); ax.set_xlabel("n_boxes"); ax.set_ylabel("images")
    plt.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_coverage(df, info, out):
    """original_split vs new split, has_no_damage per split, country vs split."""
    panels = []
    if "original_split" in df.columns and info["split"]:
        panels.append(("original_split vs new split",
                       pd.crosstab(df["original_split"], df[info["split"]]),
                       "new split"))
    if "has_no_damage" in df.columns and info["split"]:
        panels.append(("has_no_damage per split",
                       pd.crosstab(df[info["split"]], df["has_no_damage"]),
                       "has_no_damage"))
    if "country" in df.columns and info["split"]:
        panels.append(("country vs new split",
                       pd.crosstab(df["country"], df[info["split"]]),
                       "new split"))

    if not panels:
        return False

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (title, cros, legend_title) in zip(axes, panels):
        cros.plot.bar(ax=ax)
        ax.set_title(title)
        ax.set_ylabel("images"); ax.set_xlabel("")
        ax.legend(title=legend_title, fontsize=8)
        # annotate largest bars to keep the plot readable
        for c in cros.columns:
            for i, v in enumerate(cros[c]):
                if v:
                    ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return True


def resolve_image_path(raw, images_dir, data_dir):
    raw = str(raw)
    p = Path(raw)
    cands = []
    if p.is_absolute():
        cands.append(p)
    else:
        if images_dir:
            cands += [images_dir / p, images_dir / p.name]
        cands += [
            data_dir / p, data_dir / p.name,
            data_dir / "raw" / p, data_dir / "raw" / p.name,
            data_dir / "images" / p, data_dir / "images" / p.name,
            data_dir / "rdd" / p, data_dir / "rdd" / p.name,
            Path.cwd() / p, Path.cwd() / p.name,
        ]
    for c in cands:
        if c.exists():
            return c
    return None


def fig_sample_grid(df, info, images_dir, data_dir, damage_classes, n, out):
    name_col = info["name"] or info["path"]
    if name_col is None:
        print("No name/path column - skipping sample grid.")
        return False

    resolved = []
    for _, row in df.sample(n=min(n * 5, len(df)), random_state=0).iterrows():
        p = resolve_image_path(row[name_col], images_dir, data_dir)
        if p is not None:
            resolved.append((row, p))
        if len(resolved) >= n:
            break

    if not resolved:
        print(f"Could not find any images for column '{name_col}'.")
        print("  Tip: run with --images-dir /path/to/jpgs")
        return False

    cols = 4
    rows = (len(resolved) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = [axes] if rows * cols == 1 else np.array(axes).flatten()

    for ax, (row, p) in zip(axes, resolved):
        try:
            ax.imshow(Image.open(p).convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, f"unreadable:\n{p.name}", ha="center", va="center", fontsize=8)

        present = [cls for cls in damage_classes
                   if f"has_{cls}" in row and bool(row[f"has_{cls}"])]
        if not present and "has_no_damage" in row and bool(row["has_no_damage"]):
            present = ["no_damage"]
        title = p.name
        if present:
            title = " ".join(present) + "  |  " + title
        if info["split"]:
            title += f"  [{row[info['split']]}]"
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    for ax in axes[len(resolved):]:
        ax.axis("off")
    plt.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return True


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    info = detect_columns(df, DAMAGE_CLASSES)
    tables = build_tables(df, info, DAMAGE_CLASSES)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"CSV      : {csv_path}")
        print(f"Rows     : {len(df)}")
        print(f"Columns  : {list(df.columns)}")
        print(f"Detected : split={info['split']!r}  label={info['label']!r}  "
              f"path={info['path']!r}  name={info['name']!r}")
        print(f"           count_cols={info['count_cols']}")
        print(f"           has_cols={info['has_cols']}")
        print()
        for name, t in tables.items():
            print(f"--- {name} ---"); print(t); print()
    summary = buf.getvalue()
    print(summary)
    (out_dir / "summary.txt").write_text(summary)

    with pd.ExcelWriter(out_dir / "overview.xlsx") as w:
        if not tables:
            pd.DataFrame({"info": ["no tables - check CSV columns"]}).to_excel(w, sheet_name="info")
        for name, t in tables.items():
            t.to_excel(w, sheet_name=name[:31])

    images_dir = Path(args.images_dir) if args.images_dir else None
    made = []
    if "images_per_split" in tables:
        fig_split_sizes(tables["images_per_split"], out_dir / "split_sizes.png"); made.append("split_sizes.png")
    if "class_image_counts" in tables:
        fig_class_image_counts(tables["class_image_counts"], out_dir / "class_balance.png"); made.append("class_balance.png")
    if "per_split_crosstab" in tables:
        fig_per_split(tables["per_split_crosstab"], out_dir / "per_split_classes.png"); made.append("per_split_classes.png")
    if "class_box_counts" in tables:
        fig_box_counts(tables["class_box_counts"], out_dir / "boxes_per_class.png"); made.append("boxes_per_class.png")
    if "n_boxes" in df.columns:
        fig_boxes_hist(df, info["split"], out_dir / "boxes_hist.png"); made.append("boxes_hist.png")
    if fig_coverage(df, info, out_dir / "coverage.png"):
        made.append("coverage.png")
    if fig_sample_grid(df, info, images_dir, DATA_DIR, DAMAGE_CLASSES, args.n, out_dir / "sample_grid.png"):
        made.append("sample_grid.png")

    print("Saved figures:", ", ".join(made))
    print(f"Saved report to {out_dir}/")


if __name__ == "__main__":
    main()
