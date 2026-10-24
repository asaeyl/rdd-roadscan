# Fine-tuned final model (Crack, Pothole, Rut)

`model/final_model/best.pt` trained further on the local photos (train split of `local_split.csv`), 20 epochs, same settings as `model/final_model.py`, but with the 3 LOCAL classes: Crack, Pothole, Rut.

Thresholds (local validation): {'Crack': 0.13, 'Pothole': 0.25, 'Rut': 0.16}

Local test mAP@0.5 = 0.490, mAP@0.5:0.95 = 0.240.
Only the `test` photos of `local_split.csv` may be used to judge it.

Tables as csv: `tables/`. Notes:
- 3-CLASS MODEL: Crack, Pothole, Rut (the classes of the local labels). It is NOT a D00 / D10 / D20 / D40 model: the app and E4 code must use artifact.json 'classes'.
- Trained on the local TRAIN photos only, starting from model/final_model/best.pt (backbone and neck weights kept, class outputs new). Nothing from the RDD split is used here.
- Thresholds were chosen on the local VALIDATION photos; the local TEST photos were used once, for these tables.
- 'original (merged)' = model/final_model/best.pt with ITS OWN thresholds (chosen on the RDD validation split), its D00 / D10 / D20 boxes merged into Crack (duplicates removed by NMS) and D40 = Pothole, on the same local test photos. It has no Rut class, so its Rut recall is 0 by design: compare with 'macro (Crack + Pothole)' for a fair look.
- The local sets are small (see the box counts): a class with few boxes in validation / test gives an unreliable threshold / precision / recall.
- Same settings as model/final_model.py (imported from it): model size, image size, epochs, batch size, seed, Ultralytics extras.
- best.pt = epoch with the best validation score; last.pt = final epoch. artifact.json belongs to best.pt.
- For the next experiment use ONLY the 'test' photos of local_split.csv: the train and validation photos were seen during training.
