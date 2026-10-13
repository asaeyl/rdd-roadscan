"""scripts/explore_split.py

Explore the frozen split file (data/splits_*.csv) and save a full report.

Outputs (all in data/split_explore/):
  - summary.txt              text report (splits, class balance, crosstab)
  - overview.xlsx            tables (sheets: images_per_split, class_balance, per_split_crosstab)
  - split_sizes.png          bar chart: images per split
  - class_balance.png        bar chart: images per class (all splits)
  - per_split_classes.png    grouped bar chart: class balance per split
  - sample_grid.png          grid of sample images

Usage:
    python scripts/explore_split.py
    python scripts/explore_split.py --csv data/splits_12000.csv --n 16
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

try:
    from src import config as cfg
    DATA_DIR = Path(cfg.DATA_DIR)
    DAMAGE_CLASSES = list(getattr(cfg, "DAMAGE_CLASSES", []))
except Exception as e:
    print("Could not import src.config, using defaults:", e)
    DATA_DIR = Path("data")
    DAMAGE_CLASSES = ["D00", "D10", "D20", "D40"]

_csvs = sorted(DATA_DIR.glob("splits_*.csv"))
DEFAULT_CSV = _csvs[-1] if _csvs else DATA_DIR / "splits_12000.csv"
DEFAULT_OUT = DATA_DIR / "split_explore"


# ---------------------------------------------------------------- helpers

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--n", type=int, default=16, help="images in the sample grid")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    return ap.parse_args()


def detect_columns(df):
    split_col = next((c for c in df.columns
                      if c.lower() in ("split", "subset", "set", "fold")), None)
    label_col = next((c for c in df.columns
                      if c.lower() in ("class", "label", "damage", "damage_class", "category")), None)
    path_col  = next((c for c in df.columns
                      if c.lower() in ("path", "image", "image_path", "filename", "file")), None)
    return split_col, label_col, path_col


def resolve_path(raw, data_dir):
    p = Path(str(raw))
    if p.is_absolute():
        return p
    candidates = [Path.cwd() / p, data_dir / p, data_dir.parent / p]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # best guess


def build_tables(df, split_col, label_col):
    tables = {}
    if split_col:
        t = df[split_col].value_counts().sort_index().rename("images").to_frame()
        t.index.name = "split"
        tables["images_per_split"] = t
    if label_col:
        t = df[label_col].value_counts().sort_index().rename("images").to_frame()
        t.index.name = "class"
        tables["class_balance"] = t
        if split_col:
            tables["per_split_crosstab"] = pd.crosstab(df[split_col], df[label_col])
    return tables


# ---------------------------------------------------------------- figures

def fig_split_sizes(table, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    table["images"].plot.bar(ax=ax, color="#4C72B0")
    ax.set_title("Images per split")
    ax.set_ylabel("images")
    ax.set_xlabel("")
    for i, v in enumerate(table["images"]):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_class_balance(table, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    table["images"].plot.bar(ax=ax, color="#55A868")
    ax.set_title("Class balance (all splits)")
    ax.set_ylabel("images")
    ax.set_xlabel("")
    for i, v in enumerate(table["images"]):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_per_split(crosstab, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    crosstab.plot.bar(ax=ax)
    ax.set_title("Class balance per split")
    ax.set_ylabel("images")
    ax.set_xlabel("")
    ax.legend(title="class", fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_sample_grid(df, path_col, label_col, split_col, data_dir, n, out):
    sample = df.sample(n=min(n, len(df)), random_state=0)
    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = [axes] if rows * cols == 1 else axes.flatten()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        p = resolve_path(row[path_col], data_dir)
        try:
            img = Image.open(p).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, f"missing:\n{p.name}", ha="center", va="center", fontsize=8)
        title = p.name
        if label_col and label_col in row:
            title = f"{row[label_col]}  |  {title}"
        if split_col:
            title += f"  [{row[split_col]}]"
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    split_col, label_col, path_col = detect_columns(df)
    tables = build_tables(df, split_col, label_col)

    # --- text report ---
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"CSV      : {csv_path}")
        print(f"Rows     : {len(df)}")
        print(f"Columns  : {list(df.columns)}")
        print(f"Detected : split={split_col!r}  label={label_col!r}  path={path_col!r}")
        print()
        if not tables:
            print("No split / class columns detected - see 'Columns' above.")
            print()
        for name, t in tables.items():
            print(f"--- {name} ---")
            print(t)
            print()
        print("Figures:")
        for f in ("split_sizes.png", "class_balance.png",
                  "per_split_classes.png", "sample_grid.png"):
            print(f"  {out_dir / f}")
    summary = buf.getvalue()
    print(summary)
    (out_dir / "summary.txt").write_text(summary)

    # --- excel ---
    xlsx_path = out_dir / "overview.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        if not tables:
            pd.DataFrame({"info": ["no tables - check CSV columns"]}).to_excel(
                writer, sheet_name="info")
        for name, t in tables.items():
            t.to_excel(writer, sheet_name=name[:31])

    # --- figures ---
    if "images_per_split" in tables:
        fig_split_sizes(tables["images_per_split"], out_dir / "split_sizes.png")
    if "class_balance" in tables:
        fig_class_balance(tables["class_balance"], out_dir / "class_balance.png")
    if "per_split_crosstab" in tables:
        fig_per_split(tables["per_split_crosstab"], out_dir / "per_split_classes.png")
    if path_col:
        fig_sample_grid(df, path_col, label_col, split_col,
                        DATA_DIR, args.n, out_dir / "sample_grid.png")

    print(f"Saved report to {out_dir}/")


if __name__ == "__main__":
    main()
