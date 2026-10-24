import os
import shutil
import pandas as pd

METADATA_PATH = "dataset/metadata.csv"
BACKUP_PATH = "dataset/metadata_backup.csv"

# 1. Create a backup of original metadata if not exists
if not os.path.exists(BACKUP_PATH) and os.path.exists(METADATA_PATH):
    shutil.copyfile(METADATA_PATH, BACKUP_PATH)
    print(f"Backed up original metadata to {BACKUP_PATH}")

df = pd.read_csv(METADATA_PATH)
print(f"Original records: {len(df)}")

# 2. Exclude non-speech/chanting/out-of-state videos
EXCLUDE_URLS = [
    "https://www.youtube.com/watch?v=IeE8on0b9RY",  # Saptakoteshwar Udakshanti Chanting (1)
    "https://www.youtube.com/watch?v=dZ6GqP71XRY",  # Saptakoteshwar Udakshanti Chanting (2)
    "https://www.youtube.com/watch?v=xfBonMri19w",  # Ghumat music competition
    "https://www.youtube.com/watch?v=PwduGEH1MTA",  # Song audio clip
    "https://www.youtube.com/watch?v=_X3UAQqOIF0",  # Udupi / Karnataka Kharvi
    "https://www.youtube.com/watch?v=7D9d4CUorHk",  # Udupi / Karnataka Kharvi
    "https://www.youtube.com/watch?v=MjjGIvgEJ90",  # Kundapura / Karnataka Kharvi
]

df_clean = df[~df["source_url"].isin(EXCLUDE_URLS)].copy()
print(f"Records after removing out-of-distribution URLs: {len(df_clean)}")

# 3. Correct dialect & region misattributions
# Digambar Kamat (Margao candidate) -> Sashti_Hindu (South)
mask_kamat = df_clean["source_url"].str.contains("klzyjrzBlKg", na=False)
df_clean.loc[mask_kamat, "dialect"] = "Sashti_Hindu"
df_clean.loc[mask_kamat, "region"] = "South"

# Vijai Sardesai (Fatorda/Margao leader) -> Sashti_Hindu (South)
mask_vijai = df_clean["source_url"].str.contains("7EFse5aUrw8", na=False)
df_clean.loc[mask_vijai, "dialect"] = "Sashti_Hindu"
df_clean.loc[mask_vijai, "region"] = "South"

# Quepem Gaoncar -> Gawda_Kunbi (South_Interior)
mask_gauncar = df_clean["source_url"].str.contains("aqr3zMFm1n4", na=False)
df_clean.loc[mask_gauncar, "dialect"] = "Gawda_Kunbi"
df_clean.loc[mask_gauncar, "region"] = "South_Interior"

# Merge legacy Saxtti into Sashti_Christian
df_clean["dialect"] = df_clean["dialect"].replace({"Saxtti": "Sashti_Christian"})

# 4. Filter micro-fragments (< 1.5 seconds)
df_clean = df_clean[df_clean["duration_sec"] >= 1.5].copy()
print(f"Records after duration filter (>= 1.5s): {len(df_clean)}")

# 5. Speaker-disjoint stratified holdout for validation
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

# Save cleaned datasets
df_clean.to_csv("dataset/metadata_clean.csv", index=False)
train_df.to_csv("dataset/train.csv", index=False)
val_df.to_csv("dataset/val.csv", index=False)

print("\n" + "=" * 55)
print(f"Cleaned dataset saved: {len(df_clean)} total clips")
print(f"Train set: {len(train_df)} clips -> dataset/train.csv")
print(f"Val set:   {len(val_df)} clips -> dataset/val.csv")
print("=" * 55)

print("\nTrain dialect distribution:")
print(train_df["dialect"].value_counts())
print("\nValidation dialect distribution (Balanced & Speaker-Disjoint):")
print(val_df["dialect"].value_counts())
