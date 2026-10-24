import os
import zipfile
import pandas as pd
from tqdm import tqdm

ZIP_OUTPUT = "konkani_dataset_colab.zip"

train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/val.csv")

required_files = set(train_df["file_name"]).union(set(val_df["file_name"]))
print(f"Packaging {len(required_files)} audio clips for Google Colab into {ZIP_OUTPUT}...")

with zipfile.ZipFile(ZIP_OUTPUT, "w", zipfile.ZIP_DEFLATED) as zipf:
    # 1. Add CSVs
    zipf.write("dataset/train.csv", arcname="train.csv")
    zipf.write("dataset/val.csv", arcname="val.csv")
    if os.path.exists("dataset/metadata_clean.csv"):
        zipf.write("dataset/metadata_clean.csv", arcname="metadata_clean.csv")

    # 2. Add required audio files
    for fname in tqdm(required_files, desc="Compressing audio clips"):
        src_path = os.path.join("dataset/processed_wavs", fname)
        if os.path.exists(src_path):
            zipf.write(src_path, arcname=f"processed_wavs/{fname}")

zip_size_mb = os.path.getsize(ZIP_OUTPUT) / (1024 * 1024)
print(f"\nSuccessfully created {ZIP_OUTPUT} ({zip_size_mb:.1f} MB)")
