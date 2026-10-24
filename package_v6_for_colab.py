import os
import zipfile
import pandas as pd
from tqdm import tqdm

ZIP_OUTPUT = "konkani_dataset_v6_colab.zip"

train_df = pd.read_csv("dataset/train_v6.csv")
val_df = pd.read_csv("dataset/val_v6.csv")

required_files = set(train_df["file_name"]).union(set(val_df["file_name"]))
print(f"Packaging {len(required_files)} audio clips for Google Colab into {ZIP_OUTPUT}...")

with zipfile.ZipFile(ZIP_OUTPUT, "w", zipfile.ZIP_DEFLATED) as zipf:
    # 1. Add CSVs (both standard names and v6 names so Colab scripts find them automatically)
    zipf.write("dataset/train_v6.csv", arcname="train.csv")
    zipf.write("dataset/val_v6.csv", arcname="val.csv")
    zipf.write("dataset/metadata_clean_v6.csv", arcname="metadata_clean.csv")
    zipf.write("dataset/train_v6.csv", arcname="train_v6.csv")
    zipf.write("dataset/val_v6.csv", arcname="val_v6.csv")

    # 2. Add required audio files
    for fname in tqdm(required_files, desc="Compressing audio clips"):
        src_path = os.path.join("dataset/processed_wavs", fname)
        if os.path.exists(src_path):
            zipf.write(src_path, arcname=f"processed_wavs/{fname}")

zip_size_mb = os.path.getsize(ZIP_OUTPUT) / (1024 * 1024)
print(f"\n🎉 Successfully created {ZIP_OUTPUT} ({zip_size_mb:.1f} MB)")
