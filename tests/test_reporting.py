import numpy as np
import pandas as pd

from src.metrics import find_best_thresholds, metrics_table
from src.reporting import save_results


def test_metrics_and_reports(tmp_path):
    rng = np.random.default_rng(0)
    labels = (rng.random((300, 4)) < .4).astype(float)
    scores = np.clip(labels * .4 + rng.random((300, 4)) * .6, 0, 1)
    thresholds = find_best_thresholds(labels, scores)
    table = metrics_table(labels, scores, thresholds)
    assert list(table.index) == ["D00", "D10", "D20", "D40", "macro"] and table.loc["macro", "AP"] > 0.5

    xlsx, txt = save_results(tmp_path, "Test experiment", {"sheet one": table}, {"seed": 42, "split id": "abc"}, ["a note"])
    sheets = pd.read_excel(xlsx, sheet_name=None, index_col=0)
    assert set(sheets) == {"settings", "sheet one"}
    text = txt.read_text()
    assert "Test experiment" in text and "seed: 42" in text and "a note" in text
