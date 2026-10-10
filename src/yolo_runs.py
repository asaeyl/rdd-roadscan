"""Helpers shared by E2 (size / resolution) and E3 (augmentation): train-or-reuse one YOLO run, then measure it."""
from pathlib import Path

from src import config as cfg
from src import detector
from src.experiment import reuse_e1_yolo


def get_or_train_run(setup, run_name, start_weights, imgsz, extra=None, same_as_e1=False):
    """Make sure weights for `run_name` exist in setup.weights_dir/run_name. Returns that folder.
    1) already trained (e.g. after a Colab reset, weights come back with git clone) -> nothing to do
    2) `same_as_e1` and E1 weights exist -> copy them
    3) otherwise train."""
    destination = Path(setup.weights_dir) / run_name
    if (destination / "best.pt").exists() and not setup.args.retrain:
        print(f"[{run_name}] weights found -> not training again")
        return destination
    if same_as_e1 and not setup.args.retrain and reuse_e1_yolo(setup, destination):
        return destination
    print(f"[{run_name}] training: model {start_weights}, image size {imgsz}, extra settings {extra}")
    run_dir = detector.train_yolo(f"{setup.name}_{run_name}", start_weights, setup.data_yaml, setup.epochs, imgsz,
                                  cfg.BATCH_SIZE, {**(extra or {}), **setup.yolo_extra})
    return detector.collect_weights(run_dir, destination)


def measure_run(setup, run_folder, imgsz):
    """Everything we report for one trained run: box metrics on TEST, size, CPU time per image."""
    best = Path(run_folder) / "best.pt"
    test_images = setup.part("test").image_path.tolist()[:cfg.CPU_TIMING_IMAGES]
    row = detector.evaluate_boxes(best, setup.data_yaml, imgsz)
    params, size_mb = detector.model_size(best)
    row.update({"params_millions": params, "weights_MB": size_mb})
    row.update(detector.measure_cpu_speed(best, test_images, imgsz))
    return row
