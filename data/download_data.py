"""Download RDD2022 (Kaggle copy) and print where it is. Raw data is NOT stored in this repository.

    python data/download_data.py
"""
import kagglehub

path = kagglehub.dataset_download("aliabdelmenam/rdd-2022")
print("Dataset downloaded to:", path)
print("Expected folder with train/ val/ test/ :", path + "/RDD_SPLIT")
print("To use a local copy instead, set the environment variable RDD_DATA_DIR to that RDD_SPLIT folder.")
