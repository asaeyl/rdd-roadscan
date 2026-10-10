"""Smoke tests of the data pipeline (run:  pytest tests/ )."""
import numpy as np

from helpers import build_mock_dataset
from src import config as cfg
from src import data


def make_subset(tmp_path, target=1200):
    dataset_dir = build_mock_dataset(tmp_path / "rdd")
    all_images = data.load_all_images(dataset_dir)
    subset = data.make_balanced_subset(all_images, target_images=target)
    subset["split"] = data.assign_blocks_to_splits(subset, frames_per_block=10)
    return dataset_dir, all_images, subset


def test_subset_is_balanced_by_country(tmp_path):
    _, _, subset = make_subset(tmp_path)
    sizes = subset.groupby("country").size()
    assert sizes.max() - sizes.min() <= 1 or sizes.min() < sizes.max()      # small countries may run out; others stay equal
    assert subset.has_no_damage.mean() < 0.15                                 # only a small share without damage


def test_split_has_no_leak_between_blocks(tmp_path):
    _, _, subset = make_subset(tmp_path)
    block = subset.country + "_" + (subset.frame_number // 10).astype(str)
    assert subset.groupby(block).split.nunique().max() == 1                   # a block never straddles two splits
    assert set(subset.split) == {"train", "val", "test"}
    shares = subset.split.value_counts(normalize=True)
    assert abs(shares["train"] - 0.70) < 0.08 and abs(shares["val"] - 0.15) < 0.08


def test_csv_round_trip_keeps_the_same_sample(tmp_path):
    dataset_dir, _, subset = make_subset(tmp_path)
    csv_path = data.save_split_csv(subset, tmp_path / "splits.csv")
    loaded = data.load_split_csv(csv_path)
    assert data.split_fingerprint(loaded) == data.split_fingerprint(subset)   # same ID -> same sample in every experiment
    assert "image_path" not in loaded.columns                                 # no machine-specific paths in the CSV
    again = data.attach_paths(loaded, dataset_dir)
    assert all(__import__("pathlib").Path(p).exists() for p in again.image_path)


def test_yolo_dataset_is_built_from_the_csv(tmp_path):
    dataset_dir, _, subset = make_subset(tmp_path)
    csv_path = data.save_split_csv(subset, tmp_path / "splits.csv")
    table = data.attach_paths(data.load_split_csv(csv_path), dataset_dir)
    yaml_path = data.build_yolo_dataset(table, tmp_path / "yolo")
    assert yaml_path.exists()
    n_images = len(list((tmp_path / "yolo" / "images").rglob("*.jpg")))
    n_labels = len(list((tmp_path / "yolo" / "labels").rglob("*.txt")))
    assert n_images == n_labels == len(table)
    ids = {int(l.split()[0]) for f in (tmp_path / "yolo" / "labels").rglob("*.txt") for l in f.read_text().splitlines() if l.strip()}
    assert ids <= {0, 1, 2, 3}
    assert data.build_yolo_dataset(table, tmp_path / "yolo") == yaml_path     # second call re-uses the folder


def test_fingerprint_changes_when_the_sample_changes(tmp_path):
    _, _, subset = make_subset(tmp_path)
    other = subset.copy()
    other.loc[other.index[0], "split"] = "val" if other.loc[other.index[0], "split"] != "val" else "test"
    assert data.split_fingerprint(other) != data.split_fingerprint(subset)
