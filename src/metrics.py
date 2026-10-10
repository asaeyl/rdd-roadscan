"""Metrics used by ALL experiments, so every model is measured with the same ruler."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src import config as cfg


def mean_average_precision(labels, scores):
    """Mean over the 4 damage types of the average precision (AP)."""
    return float(np.mean([average_precision_score(labels[:, k], scores[:, k]) for k in range(len(cfg.DAMAGE_CLASSES))]))


def find_best_thresholds(labels, scores):
    """For each damage type, the score threshold with the best F1 on this data. Use VALIDATION data, never test."""
    candidates = np.linspace(0.05, 0.95, 91)
    best = []
    for k in range(len(cfg.DAMAGE_CLASSES)):
        f1_values = [f1_score(labels[:, k], scores[:, k] >= t, zero_division=0) for t in candidates]
        best.append(float(candidates[int(np.argmax(f1_values))]))
    return best


def metrics_table(labels, scores, thresholds):
    """AP, precision, recall, F1 per damage type + a 'macro' row (the average of the four)."""
    results = {}
    for k, name in enumerate(cfg.DAMAGE_CLASSES):
        predicted = scores[:, k] >= thresholds[k]
        results[name] = {
            "AP": average_precision_score(labels[:, k], scores[:, k]),
            "precision": precision_score(labels[:, k], predicted, zero_division=0),
            "recall": recall_score(labels[:, k], predicted, zero_division=0),
            "F1": f1_score(labels[:, k], predicted, zero_division=0),
            "threshold": thresholds[k],
            "share_of_images": labels[:, k].mean()}
    table = pd.DataFrame(results).T
    table.loc["macro"] = table.mean()
    return table
