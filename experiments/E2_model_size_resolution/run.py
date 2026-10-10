"""E2 - Detector size and input resolution (nano vs small, 640 vs 960).
Question: what is the accuracy vs speed trade-off on CPU?

    python experiments/E2_model_size_resolution/run.py
    python experiments/E2_model_size_resolution/run.py --smoke
    python experiments/E2_model_size_resolution/run.py --only yolo26s_960      # one run only

Same split, same epochs, same batch size and same augmentation as E1; ONLY model size and image size change.
Each model is tested at the image size it was trained with.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src import config as cfg
from src.experiment import common_settings, start_experiment
from src.reporting import save_results, save_scatter
from src.yolo_runs import get_or_train_run, measure_run

NAME = "E2_model_size_resolution"

# run name: (starting weights, image size, same recipe as the E1 YOLO run?)
RUNS = {
    "yolo26n_640": ("yolo26n.pt", 640, True),      # = the E1 detector (copied, not trained again)
    "yolo26n_960": ("yolo26n.pt", 960, False),
    "yolo26s_640": ("yolo26s.pt", 640, False),
    "yolo26s_960": ("yolo26s.pt", 960, False),
}


def main():
    setup = start_experiment(NAME, extra_args={"--only": {"default": "", "help": "comma-separated run names"}})
    chosen = [r for r in RUNS if not setup.args.only or r in setup.args.only.split(",")]

    rows = {}
    for run_name in chosen:
        start_weights, imgsz, same_as_e1 = RUNS[run_name]
        folder = get_or_train_run(setup, run_name, start_weights, imgsz, same_as_e1=same_as_e1 and imgsz == cfg.IMAGE_SIZE
                                  and start_weights == cfg.YOLO_MODEL)
        rows[run_name] = {"model": start_weights.replace(".pt", ""), "image_size": imgsz, **measure_run(setup, folder, imgsz)}
        print(run_name, {k: round(v, 3) if isinstance(v, float) else v for k, v in rows[run_name].items()})

    table = pd.DataFrame(rows).T
    for column in table.columns.drop(["model"]):
        table[column] = pd.to_numeric(table[column])
    table["mAP50_per_CPU_second"] = table["mAP50"] / (table["cpu_ms_total"] / 1000)

    save_scatter(table["cpu_ms_total"], table["mAP50"], table.index, setup.results_dir / "e2_accuracy_vs_cpu_time.png",
                 "E2 - accuracy vs speed (test set)", "CPU time per image (ms)", "box mAP@0.5")
    notes = ["Read the table as a trade-off: how much mAP does each extra millisecond of CPU time buy?",
             "Look at per-class AP50 too: small objects (thin cracks, D00/D10) usually profit most from the higher resolution.",
             "CPU time = preprocess + model + postprocess, batch size 1, same machine for all runs.",
             "yolo26n_640 is the E1 detector (weights copied, same recipe)."]
    xlsx, txt = save_results(setup.results_dir, "E2 - Detector size and input resolution", {"E2 summary": table},
                             common_settings(setup, runs=",".join(chosen)), notes)
    print(table.round(3)); print("saved:", xlsx, "and", txt)


if __name__ == "__main__":
    main()
