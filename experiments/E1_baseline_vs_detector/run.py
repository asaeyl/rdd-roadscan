"""E1 - Baseline classifier vs detector.
Question: what does localisation add, and is it worth the extra complexity?

    python experiments/E1_baseline_vs_detector/run.py            # full run
    python experiments/E1_baseline_vs_detector/run.py --smoke    # 2-minute test that everything works

Output (this folder):  weights/baseline/  weights/yolo/   results/results.xlsx  results/results.txt  results/*.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src import baseline, detector
from src import config as cfg
from src.experiment import common_settings, start_experiment
from src.metrics import find_best_thresholds, metrics_table
from src.reporting import save_bar_chart, save_results

NAME = "E1_baseline_vs_detector"


def main():
    setup = start_experiment(NAME)
    train_df, val_df, test_df = setup.part("train"), setup.part("val"), setup.part("test")

    # ------------------------------------------------------------------ 1. baseline classifier
    baseline_dir = setup.weights_dir / "baseline"
    if (baseline_dir / "best.pt").exists() and not setup.args.retrain:
        print("baseline weights found -> not training again")
        history = pd.read_csv(baseline_dir / "history.csv", index_col=0) if (baseline_dir / "history.csv").exists() else pd.DataFrame()
    else:
        history = baseline.train_baseline(train_df, val_df, baseline_dir, epochs=setup.epochs)
        history.to_csv(baseline_dir / "history.csv")

    model = baseline.load_model(baseline_dir / "best.pt")
    val_scores, val_labels = baseline.predict_scores(model, val_df)
    test_scores, test_labels = baseline.predict_scores(model, test_df)
    baseline_thresholds = find_best_thresholds(val_labels, val_scores)            # chosen on VALIDATION only
    baseline_results = metrics_table(test_labels, test_scores, baseline_thresholds)
    baseline_params, baseline_ms = baseline.model_cost(baseline_dir / "best.pt")

    # fitted preprocessing + thresholds are saved NEXT TO the weights (brief: "saved together")
    (baseline_dir / "artifact.json").write_text(json.dumps({
        "model": "resnet18", "classes": cfg.DAMAGE_CLASSES, "image_size": cfg.IMAGE_SIZE,
        "imagenet_mean": cfg.IMAGENET_MEAN, "imagenet_std": cfg.IMAGENET_STD,
        "thresholds": dict(zip(cfg.DAMAGE_CLASSES, baseline_thresholds)),
        "seed": cfg.SEED, "split_id": setup.split_id, "epochs": setup.epochs}, indent=2))

    # ------------------------------------------------------------------ 2. YOLO detector
    yolo_dir = setup.weights_dir / "yolo"
    if (yolo_dir / "best.pt").exists() and not setup.args.retrain:
        print("yolo weights found -> not training again")
    else:
        run_dir = detector.train_yolo(f"{NAME}_yolo", cfg.YOLO_MODEL, setup.data_yaml, setup.epochs,
                                      cfg.IMAGE_SIZE, cfg.BATCH_SIZE, setup.yolo_extra)
        detector.collect_weights(run_dir, yolo_dir)

    yolo_best = yolo_dir / "best.pt"
    box_results = detector.evaluate_boxes(yolo_best, setup.data_yaml, cfg.IMAGE_SIZE)          # what ONLY a detector has
    yolo_val_scores = detector.image_level_scores(yolo_best, val_df.image_path.tolist(), cfg.IMAGE_SIZE)
    yolo_test_scores = detector.image_level_scores(yolo_best, test_df.image_path.tolist(), cfg.IMAGE_SIZE)
    yolo_thresholds = find_best_thresholds(val_labels, yolo_val_scores)           # again chosen on VALIDATION only
    yolo_results = metrics_table(test_labels, yolo_test_scores, yolo_thresholds)
    yolo_params, yolo_size_mb = detector.model_size(yolo_best)
    yolo_speed = detector.measure_cpu_speed(yolo_best, test_df.image_path.tolist()[:cfg.CPU_TIMING_IMAGES], cfg.IMAGE_SIZE)

    (yolo_dir / "artifact.json").write_text(json.dumps({
        "model": cfg.YOLO_MODEL, "classes": cfg.DAMAGE_CLASSES, "image_size": cfg.IMAGE_SIZE,
        "image_level_thresholds": dict(zip(cfg.DAMAGE_CLASSES, yolo_thresholds)),
        "seed": cfg.SEED, "split_id": setup.split_id, "epochs": setup.epochs}, indent=2))

    # ------------------------------------------------------------------ 3. the E1 table
    def summary_row(results, params, ms, box_map=np.nan):
        row = {f"AP_{n}": results.loc[n, "AP"] for n in cfg.DAMAGE_CLASSES}
        row.update({"mean_AP": results.loc["macro", "AP"], "macro_F1": results.loc["macro", "F1"],
                    "params_millions": params, "CPU_ms_per_image": ms, "box_mAP@0.5": box_map})
        return row

    prior_row = {f"AP_{n}": test_labels[:, k].mean() for k, n in enumerate(cfg.DAMAGE_CLASSES)}
    prior_row.update({"mean_AP": test_labels.mean(axis=0).mean(), "macro_F1": np.nan, "params_millions": 0,
                      "CPU_ms_per_image": 0, "box_mAP@0.5": np.nan})
    e1_table = pd.DataFrame({
        "Prior (class share only)": prior_row,
        "Baseline ResNet-18": summary_row(baseline_results, baseline_params, baseline_ms),
        "YOLO detector": summary_row(yolo_results, yolo_params, yolo_speed["cpu_ms_total"], box_results["mAP50"]),
    }).T

    save_bar_chart(e1_table[[f"AP_{n}" for n in cfg.DAMAGE_CLASSES]].T, setup.results_dir / "e1_ap_per_class.png",
                   "E1 - image-level AP per damage type (test set)", "AP")
    delta = e1_table.loc["YOLO detector", "mean_AP"] - e1_table.loc["Baseline ResNet-18", "mean_AP"]
    notes = [f"Mean AP difference (YOLO - baseline) on 'is damage type X in this image?': {delta:+.3f}",
             "If this difference is small, localisation adds little for the yes/no question; its value is then in what only boxes give:",
             "box size (severity), number of instances, and crops for the duplicate finder.",
             f"Cost: baseline {baseline_ms:.0f} ms vs YOLO {yolo_speed['cpu_ms_total']:.0f} ms per image on CPU; "
             f"{baseline_params:.1f} M vs {yolo_params:.1f} M parameters; YOLO also needs box annotations.",
             "YOLO image-level score = highest confidence among the boxes of that damage type (0 if none).",
             "Thresholds for both models were chosen on the validation split, never on test."]
    tables = {"E1 summary": e1_table, "baseline per class": baseline_results, "yolo image-level per class": yolo_results,
              "yolo box metrics": pd.DataFrame([box_results], index=["YOLO (test)"]), "baseline training": history}
    xlsx, txt = save_results(setup.results_dir, "E1 - Baseline classifier vs detector", tables,
                             common_settings(setup, yolo_model=cfg.YOLO_MODEL), notes)
    print(e1_table.round(3)); print("saved:", xlsx, "and", txt)


if __name__ == "__main__":
    main()
