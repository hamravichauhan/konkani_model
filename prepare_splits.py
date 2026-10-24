import os
import pandas as pd

METADATA_PATH = "dataset/metadata_clean.csv"
if not os.path.exists(METADATA_PATH):
    METADATA_PATH = "dataset/metadata.csv"

df_clean = pd.read_csv(METADATA_PATH)

# Speaker-disjoint stratified holdout for validation
HOLD_OUT_SPEAKERS = {
    "Antruz": ["spk_OSRE5D"],                  # 42 clips
    "Bardez": ["spk_wMY0xH"],                  # 31 clips
    "Canacona": ["spk_omJx5e"],                # 20 clips
    "Gawda_Kunbi": ["spk_4tDQjH"],             # 44 clips
    "Pednekar": ["spk_nwP8Qp", "spk_Lsll6M"],  # 24 clips
    "Sashti_Christian": ["spk_8z6_8E"],        # 61 clips
    "Sashti_Hindu": ["spk_7EFse5"],            # 33 clips
    "Sattarkar": ["spk_AgPpQB"],               # 30 clips
}

val_records = []
train_records = []

for dialect, group in df_clean.groupby("dialect"):
    val_spks = HOLD_OUT_SPEAKERS.get(dialect, [])
    val_part = group[group["speaker_id"].isin(val_spks)]
    train_part = group[~group["speaker_id"].isin(val_spks)]
    val_records.append(val_part)
    train_records.append(train_part)

train_df = pd.concat(train_records).reset_index(drop=True)
val_df = pd.concat(val_records).reset_index(drop=True)

train_df.to_csv("dataset/train.csv", index=False)
val_df.to_csv("dataset/val.csv", index=False)

print("\n" + "=" * 55)
print(f"Train set: {len(train_df)} clips -> dataset/train.csv")
print(f"Val set:   {len(val_df)} clips -> dataset/val.csv")
print("=" * 55)

print("\nTrain dialect distribution:")
print(train_df["dialect"].value_counts())
print("\nValidation dialect distribution:")
print(val_df["dialect"].value_counts())
