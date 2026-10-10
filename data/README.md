# data/

* `splits_8000.csv` – the **frozen balanced 8,000-image split** (image names, labels, train/val/test). Created once by
  `python scripts/make_split.py`, then used by **every** experiment (E1, E2, E3, ...). Do not regenerate it.
* `download_data.py` – downloads RDD2022 (Kaggle copy `aliabdelmenam/rdd-2022`). Raw images are not stored in Git.
* `class_id_check.png` – picture used to verify that class ids 0-3 mean D00, D10, D20, D40.

Column meaning of `splits_8000.csv`: `name` (file name without extension, contains country and frame number),
`original_split` (folder in the Kaggle copy; used only to find the file), `split` (**our** leak-free split),
`has_Dxx` / `count_Dxx` (is the damage type present / how many boxes), `has_no_damage`.

Licence and source: check the licence on the RDD2022 source page (and the Kaggle copy) and state it in the main README.
