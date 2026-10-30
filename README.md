# RoadScan – road-damage detection and report triage (Track A)


## Problem and motivation
Municipalities receive many citizen reports of road damage, often several for the same pothole, and have no consistent way
to decide what to repair first. This repository contains the data pipeline and the experiments for the detector part:
which damage is in a photo, where it is, and how it compares with a simple classifier baseline.
It also contains the **final model** with its confidence thresholds, an **external evaluation** on photos from other sources,
a **duplicate-candidate pilot** (can the same damage be found again?), a **fine-tuning** on local Saudi photos, and a small
**Streamlit demo app**.

## Data
* **RDD2022** (Kaggle copy `aliabdelmenam/rdd-2022`), classes **D00, D10, D20, D40**.
* We use a **balanced 12,000-image subset** (equal images per country, damage types as equal as possible, ~8% images without damage).
* **Leak-free split 70/15/15**: images with nearby frame numbers (same drive, near-identical) are kept together in blocks of 50.
  The original Kaggle split is random over consecutive frames and is **not** used.
* The split is created **once** (`data/splits_12000.csv`) and shared by **all experiments**. Every results file prints the same
  *split id*, which proves they used the identical sample.

### Local and external photos (used from E4 on)
| Set | Photos | Labels | Used for |
|---|---|---|---|
| Local photos (first E4 run) | see `experiments/E4_local_photos/results/` | RDD classes | testing the final model, no training |
| Set 1: team-captured mobile-phone photos (Saudi) | 28 | RDD classes (D00, D10, D20, D40) | external evaluation (RDD-Data) |
| Set 2: Saudi road-damage dataset (Roboflow `saudird`, CC BY 4.0) | 217 | Crack, Pothole, Rut | held-out test (RDD-test) |
| Saudi road-damage dataset, training part | 820 | Crack, Pothole, Rut | fine-tuning: 574 train / 123 val / 123 test, split by source photo |

**The photos themselves are not in this repository (it is public).** Pictures go to a folder outside the repo (a zip in Colab);
only numbers (xlsx, txt, csv, charts without photos) are committed.

### Label classes: the same number means different things
| id | RDD labels (final model) | Saudi labels (fine-tuned model) |
|---|---|---|
| 0 | D00 – longitudinal crack | Crack |
| 1 | D10 – transverse crack | Pothole |
| 2 | D20 – alligator crack | Rut |
| 3 | D40 – pothole | – |

Reading Saudi labels as RDD labels (or the other way round) gives numbers that look real but are wrong. Every script that reads
labels therefore checks the label set. To compare the two models fairly, the final model's classes are merged:
**D00 + D10 + D20 → Crack, D40 → Pothole** (duplicate boxes on one crack removed by NMS). The final model has no Rut class.

## Repository layout
```
README.md  requirements.txt  .gitignore  .env.example  packages.txt
data/            splits_12000.csv (the frozen split) · download_data.py · README.md
scripts/         make_split.py (run once) · check_class_ids.py
src/             config.py · data.py · metrics.py · baseline.py · detector.py · yolo_runs.py · experiment.py · reporting.py
experiments/
  E1_baseline_vs_detector/    run.py · results/ (xlsx, txt, png) · weights/baseline · weights/yolo
  E2_model_size_resolution/   run.py · results/ · weights/<one folder per model>
  E3_augmentation/            run.py · results/ · weights/<one folder per setting>
  E4_local_photos/            run.py · results/ (first run) · RDD-Data/result · RDD-test/result   (final model on external photos)
  E5_Duplicate_finder/        run.py · results/                                                    (duplicate-candidate pilot)
  E6_yolo_generations/        run.py · results/ · weights/                                         (YOLO generations)
model/           final_model.py (creates the final model) · final_model/ (best.pt, artifact.json, results.xlsx)
fine-tuning/     final_model.py (fine-tunes best.pt on the local photos) · final_model/ (best.pt, last.pt, artifact.json,
                 local_split.csv, results.xlsx, tables/, plots/, training/) · label_check/
more_experment/
  E4_local_photos/            run.py · RDD-Data/result · RDD-test/result      (fine-tuned model on external photos)
app/             app.py · requirements.txt                                    (Streamlit demo)
notebooks/       00_colab_runner.ipynb  (runs everything on Colab and pushes to GitHub)
tests/           smoke tests of the data pipeline (pytest)
```

## How to run (copy-paste)
```bash
pip install -r requirements.txt
python scripts/check_class_ids.py                       # look at data/class_id_check.png: id 3 must show potholes
python scripts/make_split.py                            # ONCE: creates data/splits_12000.csv
python experiments/E1_baseline_vs_detector/run.py       # add --smoke for a 2-minute test
python experiments/E2_model_size_resolution/run.py
python experiments/E3_augmentation/run.py
python experiments/E6_yolo_generations/run.py
python model/final_model.py                             # final model + confidence thresholds -> model/final_model/
pytest tests/                                           # smoke tests
```
Each experiment writes `results/results.xlsx` (one sheet per table + settings), `results/results.txt` (same content as text)
and a chart, and saves its weights in `weights/` (`best.pt` = best epoch on validation, `last.pt` = final epoch,
plus `artifact.json` with thresholds, class names, image size, seed and split id).
If weights already exist the experiment does **not** train again, it only re-evaluates (`--retrain` forces training).
Set `RDD_DATA_DIR` to use a local copy of the dataset, `RDD_WORK_DIR` for big temporary files.

### External evaluation of the final model (E4, no training)
A photo folder has `images/` and `labels/` (YOLO format: `class_id x_center y_center width height`, values 0–1; an **empty** label file means no damage).
```bash
python experiments/E4_local_photos/run.py --local-dir /content/rdd_local          # labels = RDD classes
python experiments/E4_local_photos/run.py --local-dir /content/set2_photos --no-labels   # detections only (labels in another class set)
```
It draws the detections (photos with damage detected / no damage detected / mistakes), and with labels reports precision, recall, F1 and AP per damage type.

### Duplicate-candidate pilot (E5, no training)
```bash
python experiments/E5_Duplicate_finder/run.py --local-dir /content/rdd_test       # prepared dataset: gallery crops, altered queries, pairs.csv
```
Compares Perceptual Hash, ResNet50 (ImageNet) and DINOv2 (ViT-S/14): Recall@1, Recall@5, MRR, and pair matching (precision, recall, F1, ROC-AUC).
The positive pairs are synthetic views of the same crop, not two real photos of the same damage.

### Fine-tuning on the local Saudi photos
The fine-tuned model starts from `model/final_model/best.pt` and uses the **same settings** as `model/final_model.py`
(imported, not copied: model size, 640 px, 20 epochs, batch size, seed, training settings, threshold rule). Only the data and the classes change.
```bash
python fine-tuning/final_model.py --local-dir /content/local_photos --check-only   # BEFORE training: labels vs model classes
python fine-tuning/final_model.py --local-dir /content/local_photos --smoke        # 1 epoch on 5% of the data
python fine-tuning/final_model.py --local-dir /content/local_photos                # the real run
python fine-tuning/final_model.py --local-dir /content/local_photos --reuse        # thresholds + evaluation only
```
* The result is a **3-class model (Crack, Pothole, Rut)**. Class names and thresholds are in `fine-tuning/final_model/artifact.json`.
* The photos are split once into train / validation / test **by source photo**; the split is in `fine-tuning/final_model/local_split.csv`.
  Only the `test` photos (and photos the model never saw) may be used to judge it.
* Thresholds are chosen on the local validation photos; the local test photos are used once.

### Testing the fine-tuned model (more_experment/E4_local_photos)
```bash
python more_experment/E4_local_photos/run.py --local-dir /content/set2_photos --label-set local --name RDD-test --out-dir /content/e4ft_output_test
python more_experment/E4_local_photos/run.py --local-dir /content/set1_photos --label-set rdd   --name RDD-Data --out-dir /content/e4ft_output_data
```
* `--label-set local`: labels are Crack / Pothole / Rut. `--label-set rdd`: old labels, merged into Crack / Pothole (Rut is not judged: these labels have no Rut).
* Photos used for the fine-tuning (train / validation) are **left out automatically**.
* Each run also evaluates the original final model on the same photos (classes merged) and reports the change.
* Numbers go to `more_experment/E4_local_photos/<name>/result/` (xlsx, txt, `tables/*.csv`, chart).

### Demo app (Streamlit)
```bash
streamlit run app/app.py
```
The app uses `model/final_model/best.pt` and the thresholds in its `artifact.json` (image size and one confidence threshold per damage type).
It is a research prototype: the suggested priority levels are a simple heuristic on damage type and box size, **not** a verified safety assessment.
The app does **not** use the fine-tuned model yet (different classes).
Streamlit Cloud needs `packages.txt` in the repository root (`libgl1`, `libglib2.0-0`) and `app/requirements.txt`; choose Python 3.12 or 3.13 when deploying.

## What is identical in all experiments, and what differs
Identical: split, image size 640, batch size 16, 20 epochs, seed, metrics, thresholds chosen on validation only, CPU timing machine.
Differs on purpose: E1 – the labels used (image vs boxes); E2 – model size and image size; E3 – one augmentation setting per run; E6 – the YOLO generation.
E2 and E3 re-use the E1 detector for the "nano / 640 / default augmentation" run instead of training it again.
E4 and E5 train nothing: E4 only applies the final model to external photos, E5 only compares ready-made image embeddings.
The fine-tuning keeps every setting of the final model; it changes the data (local photos), the classes (Crack, Pothole, Rut) and the starting weights (`best.pt`).

## Results summary
_(fill in from `experiments/*/results/results.txt`, `fine-tuning/final_model/results.xlsx` and `more_experment/E4_local_photos/*/result/` after the runs)_

| Experiment | Main finding |
|---|---|
| E1 baseline vs detector | ResNet-18 mean AP = 0.900; YOLO = 0.889: YOLO box mAP@0.5 = 0.526 |
| E2 size / resolution | YOLO26s 640 achieved mAP@0.5:0.95 = 0.310 with 304.8 ms/image |
| E3 augmentation | 30° rotation reduced mAP@0.5 from 0.523 to 0.474 |
| E6 YOLO generations | YOLO26n led the five nano models with mAP@0.5 0.523 and mAP@0.5:0.95 0.293; it had 2.535M parameters and 5.450 MB weights. |
| Final model and thresholds | mAP@0.5 = 0.591; mAP@0.5:0.95 = 0.336 |
| E4 external images (final model) | |
| E5 duplicate candidates | ResNet50 Recall@1 = 0.923; MRR = 0.953|
| E6-7:  External evaluation | Original model macro F1 = 0.019 on 28 images and 0.055 on SaudiRD |
| Local fine-tuning | Shared-class macro F1 on 217 images increased from 0.170 → 0.495|

## Limitations and future work
* The subset is balanced, not the natural class distribution (cracks dominate in reality).
* The external sets are small (28 and 217 photos; the 123-photo fine-tuning test has about 19 pothole boxes): per-class precision and recall can change a lot with one photo.
* The final model and the fine-tuned model use **different class sets** (D00 / D10 / D20 / D40 vs Crack / Pothole / Rut). Merging D00, D10 and D20 into Crack is a simplification, and the final model cannot detect Rut at all, so comparisons are made on Crack and Pothole.
* Label styles differ between datasets (for example one large box around a cracked area vs several small boxes); thin cracks easily fall below the overlap threshold even when they are found.
* Fine-tuning on Saudi photos can improve results on similar photos and reduce results on others (Set 1 checks this); Set 2 results are likely optimistic if photos come from the same roads as the training photos.
