"""E3 - Augmentation settings.
Question: which augmentations actually help road-damage detection, and which are physically unrealistic for road images?

    python experiments/E3_augmentation/run.py
    python experiments/E3_augmentation/run.py --smoke
    python experiments/E3_augmentation/run.py --only no_mosaic,no_hsv

Same split, model (yolo26n), image size, epochs and batch size as E1. ONE augmentation setting changes per run,
compared with the default run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src import config as cfg
from src.experiment import common_settings, start_experiment
from src.reporting import save_bar_chart, save_results
from src.yolo_runs import get_or_train_run, measure_run

NAME = "E3_augmentation"

# run name: (what changes, Ultralytics settings, our HYPOTHESIS about realism - confirm or reject it in your report)
VARIANTS = {
    "default": ("nothing (YOLO defaults)", {},
                "reference run = E1 detector"),
    "no_mosaic": ("mosaic off", {"mosaic": 0.0},
                  "mosaic glues 4 photos together: not a real road scene, but adds context variety"),
    "no_hsv": ("colour jitter off (hue/saturation/value)", {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0},
               "brightness/saturation changes are realistic (light, weather); strong hue shifts are not (blue asphalt)"),
    "no_fliplr": ("horizontal flip off", {"fliplr": 0.0},
                  "left-right mirror is realistic (traffic side aside)"),
    "flipud_on": ("vertical flip ON", {"flipud": 0.5},
                  "UNREALISTIC: the road would be upside down, the camera never sees that"),
    "rotate_30": ("rotation up to 30 degrees", {"degrees": 30.0},
                  "UNREALISTIC for a dashcam: the horizon is level; small tilts (a few degrees) are fine"),
}


def main():
    setup = start_experiment(NAME, extra_args={"--only": {"default": "", "help": "comma-separated run names"}})
    chosen = [v for v in VARIANTS if not setup.args.only or v in setup.args.only.split(",")]
    if "default" not in chosen:
        chosen.insert(0, "default")                                       # needed as the reference

    rows = {}
    for run_name in chosen:
        change, extra, hypothesis = VARIANTS[run_name]
        folder = get_or_train_run(setup, run_name, cfg.YOLO_MODEL, cfg.IMAGE_SIZE, extra=extra, same_as_e1=False)
        rows[run_name] = {"what changes": change, "our hypothesis": hypothesis, **measure_run(setup, folder, cfg.IMAGE_SIZE)}
        print(run_name, round(rows[run_name]["mAP50"], 3))

    table = pd.DataFrame(rows).T
    for column in table.columns.drop(["what changes", "our hypothesis"]):
        table[column] = pd.to_numeric(table[column])
    table["delta_mAP50_vs_default"] = table["mAP50"] - table.loc["default", "mAP50"]
    for name in cfg.DAMAGE_CLASSES:
        table[f"delta_AP50_{name}"] = table[f"AP50_{name}"] - table.loc["default", f"AP50_{name}"]

    save_bar_chart(table[["mAP50"]], setup.results_dir / "e3_map50_by_variant.png", "E3 - box mAP@0.5 by augmentation setting (test set)", "mAP@0.5")
    notes = ["Each row changes ONE thing relative to 'default'. A positive delta means that the change helped.",
             "With one run per setting, differences of about 0.01 mAP may be noise (seed, run-to-run variation): say so in the report.",
             "The 'our hypothesis' column is a hypothesis written BEFORE training - confirm or reject it with the numbers.",
             "A setting can be physically unrealistic and still help (regularisation); report what you measured."]
    xlsx, txt = save_results(setup.results_dir, "E3 - Augmentation settings", {"E3 summary": table},
                             common_settings(setup, runs=",".join(chosen)), notes)
    print(table[["what changes", "mAP50", "delta_mAP50_vs_default"]].round(3)); print("saved:", xlsx, "and", txt)


if __name__ == "__main__":
    main()
