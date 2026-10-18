# Final model

YOLO26s, 640 px, trained for 20 epochs on the project split. Weights `best.pt` + `artifact.json` (confidence thresholds per damage type, image size, split id).

Load it from Python:
```python
import importlib.util
spec = importlib.util.spec_from_file_location('final_model', 'model/final_model.py')
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
detector = fm.RoadDamageDetector()
print(detector.predict('photo.jpg'))
```

Thresholds (validation): {'D00': 0.25, 'D10': 0.28, 'D20': 0.24, 'D40': 0.13}

Test mAP@0.5 = 0.591, mAP@0.5:0.95 = 0.336. Details: `results.txt`, `results.xlsx`, `f1_vs_confidence.png`, `plots/`, `training/`.
