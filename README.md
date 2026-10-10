# RoadScan – road-damage detection and report triage (Track A)

**Team:** _(names)_  |  **Track:** A – road damage (RDD2022)  |  **Seed:** 42 everywhere

## Problem and motivation
Municipalities receive many citizen reports of road damage, often several for the same pothole, and have no consistent way
to decide what to repair first. This repository contains the data pipeline and the experiments for the detector part:
which damage is in a photo, where it is, and how it compares with a simple classifier baseline.

## Data
* **RDD2022** (Kaggle copy `aliabdelmenam/rdd-2022`), classes **D00, D10, D20, D40**. Licence: _(copy it from the source page)_.
* We use a **balanced 8,000-image subset** (equal images per country, damage types as equal as possible, ~8% images without damage).
* **Leak-free split 70/15/15**: images with nearby frame numbers (same drive, near-identical) are kept together in blocks of 50.
  The original Kaggle split is random over consecutive frames and is **not** used.
* The split is created **once** (`data/splits_8000.csv`) and shared by **all experiments**. Every results file prints the same
  *split id*, which proves they used the identical sample.

## Repository layout
```
README.md  requirements.txt  .gitignore  .env.example
data/            splits_8000.csv (the frozen split) · download_data.py · README.md
scripts/         make_split.py (run once) · check_class_ids.py
src/             config.py · data.py · metrics.py · baseline.py · detector.py · yolo_runs.py · experiment.py · reporting.py
experiments/
  E1_baseline_vs_detector/    run.py · results/ (xlsx, txt, png) · weights/baseline · weights/yolo
  E2_model_size_resolution/   run.py · results/ · weights/<one folder per model>
  E3_augmentation/            run.py · results/ · weights/<one folder per setting>
artifacts/       weights of the final system (later)
notebooks/       00_colab_runner.ipynb  (runs everything on Colab and pushes to GitHub)
tests/           smoke tests of the data pipeline (pytest)
```

## How to run (copy-paste)
```bash
pip install -r requirements.txt
python scripts/check_class_ids.py                       # look at data/class_id_check.png: id 3 must show potholes
python scripts/make_split.py                            # ONCE: creates data/splits_8000.csv
python experiments/E1_baseline_vs_detector/run.py       # add --smoke for a 2-minute test
python experiments/E2_model_size_resolution/run.py
python experiments/E3_augmentation/run.py
pytest tests/                                           # smoke tests
```
Each experiment writes `results/results.xlsx` (one sheet per table + settings), `results/results.txt` (same content as text)
and a chart, and saves its weights in `weights/` (`best.pt` = best epoch on validation, `last.pt` = final epoch,
plus `artifact.json` with thresholds, class names, image size, seed and split id).
If weights already exist the experiment does **not** train again, it only re-evaluates (`--retrain` forces training).
Set `RDD_DATA_DIR` to use a local copy of the dataset, `RDD_WORK_DIR` for big temporary files.

## What is identical in all experiments, and what differs
Identical: split, image size 640, batch size 16, 20 epochs, seed, metrics, thresholds chosen on validation only, CPU timing machine.
Differs on purpose: E1 – the labels used (image vs boxes); E2 – model size and image size; E3 – one augmentation setting per run.
E2 and E3 re-use the E1 detector for the "nano / 640 / default augmentation" run instead of training it again.

## Results summary
_(fill in from `experiments/*/results/results.txt` after the runs)_

| Experiment | Main finding |
|---|---|
| E1 baseline vs detector | |
| E2 size / resolution | |
| E3 augmentation | |

## Limitations and future work
* The subset is balanced, not the natural class distribution (cracks dominate in reality).
* One run per setting (one seed): small differences may be noise.
* Baseline and YOLO use different pretraining (ImageNet vs COCO) and different aspect-ratio handling (stretch vs padding).
* Frame numbers are only a proxy for "same drive"; a perceptual-hash check reports leftover near-duplicates (`scripts/make_split.py`).
