"""E6 - Compare YOLO generations (v5, v8, v10, yolo26).

Question: which YOLO generation gives the best accuracy vs CPU-speed trade-off on road damage?

    python experiments/E6_model_versions/run.py
    python experiments/E6_model_versions/run.py --smoke
    python experiments/E6_model_versions/run.py --only yolov8n

Same split, model size (nano), image size, epochs, batch size and augmentation as E1.
ONLY the model generation changes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src import config as cfg
from src.experiment import common_settings, start_experiment
from src.reporting import save_bar_chart, save_results, save_scatter
from src.yolo_runs import get_or_train_run, measure_run

NAME = "E6_model_versions"

# run name: (starting weights, image size, same recipe as the E1 YOLO run?)
# If your Ultralytics version uses different filenames (e.g. yolov5nu.pt),
# change the strings below - the column "model" in the results will follow them.
RUNS = {
    "yolov5n":  ("yolov5nu.pt", cfg.IMAGE_SIZE, False),
    "yolov8n":  ("yolov8n.pt",  cfg.IMAGE_SIZE, False),
    "yolov10n": ("yolov10n.pt", cfg.IMAGE_SIZE, False),
    "yolo26n":  ("yolo26n.pt",  cfg.IMAGE_SIZE, False),
}


def main():
    setup = start_experiment(NAME, extra_args={"--only": {"default": "", "help": "comma-separated run names"}})
    chosen = [r for r in RUNS if not setup.args.only or r in setup.args.only.split(",")]

    rows = {}
    for run_name in chosen:
        start_weights, imgsz, same_as_e1 = RUNS[run_name]
        folder = get_or_train_run(
            setup, run_name, start_weights, imgsz,
            same_as_e1=same_as_e1 and imgsz == cfg.IMAGE_SIZE
                       and start_weights == cfg.YOLO_MODEL,
        )
        rows[run_name] = {
            "model": start_weights.replace(".pt", ""),
            "image_size": imgsz,
            **measure_run(setup, folder, imgsz),
        }
        print(run_name, {k: round(v, 3) if isinstance(v, float) else v
                         for k, v in rows[run_name].items()})

    table = pd.DataFrame(rows).T
    for column in table.columns.drop(["model"]):
        table[column] = pd.to_numeric(table[column])
    table["mAP50_per_CPU_second"] = table["mAP50"] / (table["cpu_ms_total"] / 1000)

    save_bar_chart(
        table[["mAP50"]],
        setup.results_dir / "e6_map50_by_generation.png",
        "E6 - box mAP@0.5 by YOLO generation (test set)",
        "mAP@0.5",
    )
    save_scatter(
        table["cpu_ms_total"], table["mAP50"], table.index,
        setup.results_dir / "e6_accuracy_vs_cpu_time.png",
        "E6 - accuracy vs CPU speed by generation (test set)",
        "CPU time per image (ms)", "box mAP@0.5",
    )

    notes = [
        "One run per YOLO generation, nano size, same image size, epochs, batch and augmentation as E1.",
        "Read the table as a trade-off: which generation buys the most mAP for a given CPU cost?",
        "Look at per-class AP50 too: newer generations may help thin cracks (D00/D10) more than potholes (D40).",
        "CPU time = preprocess + model + postprocess, batch size 1, same machine for all runs.",
        "One run per generation is weak evidence: differences below ~0.01 mAP may be noise.",
    ]
    xlsx, txt = save_results(
        setup.results_dir,
        "E6 - YOLO generations",
        {"E6 summary": table},
        common_settings(setup, runs=",".join(chosen)),
        notes,
    )
    print(table.round(3))
    print("saved:", xlsx, "and", txt)


if __name__ == "__main__":
    main()
