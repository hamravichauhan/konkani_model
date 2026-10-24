import os
import re
import pandas as pd
import torch
import torchaudio
from yt_dlp import YoutubeDL

# -------------------------------------------------------------
# 1. LOAD ALREADY PROCESSED VIDEOS (PREVENTS RE-DOWNLOAD)
# -------------------------------------------------------------
metadata_csv_path = "dataset/metadata.csv"
existing_video_ids = set()

if os.path.exists(metadata_csv_path):
    existing_df = pd.read_csv(metadata_csv_path)
    if "source_url" in existing_df.columns:
        for url in existing_df["source_url"].dropna().unique():
            if "watch?v=" in url:
                existing_video_ids.add(url.split("watch?v=")[-1].strip())
    print(f"Found {len(existing_video_ids)} already processed videos in metadata. These will be skipped automatically.\n")

# -------------------------------------------------------------
# 2. EXPANDED TALUKA & COMMUNITY SEARCH TARGETS
# -------------------------------------------------------------
TALUKA_SEARCH_CONFIG = [
    # --- 1. INDIGENOUS / TRIBAL (Gawda, Kunbi, Velip) ---
    {
        "dialect": "Gawda_Kunbi",
        "region": "South_Interior",
        "query": "Goan Gawda Kunbi traditional Konkani talk documentary",
        "count": 4
    },
    {
        "dialect": "Gawda_Kunbi",
        "region": "South_Interior",
        "query": "Velip Konkani interview Canacona Goa",
        "count": 3
    },

    # --- 2. GHAT BORDERLANDS (Sattari / Valpoi) ---
    {
        "dialect": "Sattarkar",
        "region": "North_East",
        "query": "Valpoi Sattari local interview Konkani news report",
        "count": 4
    },
    {
        "dialect": "Sattarkar",
        "region": "North_East",
        "query": "Sattari Goa village locals talking Konkani",
        "count": 3
    },

    # --- 3. COASTAL COMMUNITY (Bhandari / Fisherfolk) ---
    {
        "dialect": "Bhandari",
        "region": "North_Central",
        "query": "Goan fishermen Konkani interview Kharvi coastal",
        "count": 4
    },

    # --- 4. SOUTH GOA (Sashti Christian vs Sashti Hindu) ---
    {
        "dialect": "Sashti_Christian",
        "region": "South",
        "query": "Margao Benaulim tiatr artist interview Konkani",
        "count": 4
    },
    {
        "dialect": "Sashti_Hindu",
        "region": "South",
        "query": "Cuncolim Margao local Hindu temple festival interview Konkani",
        "count": 4
    },

    # --- 5. FAR NORTH / BORDERLAND (Pednekar & Kudalkar) ---
    {
        "dialect": "Pednekar",
        "region": "Far_North",
        "query": "Pernem Goa village interview Konkani news",
        "count": 3
    },
    {
        "dialect": "Kudalkar_Border",
        "region": "Far_North",
        "query": "Pernem Goa border local dialect interview Konkani Malvani",
        "count": 4
    },

    # --- 6. FAR SOUTH (Canacona) ---
    {
        "dialect": "Canacona",
        "region": "Far_South",
        "query": "Canacona Konkani ground report local people",
        "count": 3
    },

    # --- 7. CENTRAL & NORTH BALANCING (Antruz & Bardez) ---
    {
        "dialect": "Antruz",
        "region": "Central",
        "query": "Doordarshan Goa Konkani interview Priol Ponda",
        "count": 4
    },
    {
        "dialect": "Bardez",
        "region": "North",
        "query": "Aldona Siolim local Konkani interview Goa",
        "count": 4
    }
]

RAW_DIR = "dataset/raw_temp"
PROCESSED_DIR = "dataset/processed_wavs"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -------------------------------------------------------------
# 3. LOAD SILERO VAD FOR CHUNKING
# -------------------------------------------------------------
print("Loading Silero VAD model...")
vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True
)
get_speech_timestamps, _, read_audio, _, _ = utils

# -------------------------------------------------------------
# 4. SEARCH & HARVEST LOGIC
# -------------------------------------------------------------
new_metadata_records = []

def filter_already_downloaded(info_dict, *, incomplete):
    """Callback hook: Skips any video ID that already exists in metadata.csv"""
    video_id = info_dict.get("id")
    if video_id in existing_video_ids:
        return f"Video {video_id} is already in the dataset. Skipping."
    return None

def search_and_download(search_query, max_results, dialect_info):
    search_target = f"ytsearch{max_results}:{search_query}"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(RAW_DIR, "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "postprocessor_args": [
            "-ar", "16000",
            "-ac", "1"
        ],
        "quiet": False,
        "no_warnings": True,
        "ignoreerrors": True,
        "match_filter": filter_already_downloaded,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios"]
            }
        }
    }

    print(f"\n=======================================================")
    print(f"Targeting: {dialect_info['dialect']} ({dialect_info['region']})")
    print(f"Executing: {search_target}")
    print(f"=======================================================")

    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(search_target, download=True)
        except Exception as e:
            print(f"Download/Search failed for '{search_query}': {e}")
            return

        if not info or "entries" not in info:
            return

        for entry in info["entries"]:
            if not entry:
                continue

            video_id = entry.get("id")
            title = entry.get("title", "Unknown")
            raw_wav_path = os.path.join(RAW_DIR, f"{video_id}.wav")

            if not os.path.exists(raw_wav_path):
                continue

            print(f"\n[VAD Chunking] Processing: {title[:45]}... (ID: {video_id})")

            try:
                wav = read_audio(raw_wav_path, sampling_rate=16000)
                speech_timestamps = get_speech_timestamps(
                    wav,
                    vad_model,
                    sampling_rate=16000,
                    min_speech_duration_ms=3000,
                    max_speech_duration_s=8.0,
                    min_silence_duration_ms=400
                )

                speaker_tag = f"spk_{video_id[:6]}"
                saved_chunks = 0

                for idx, ts in enumerate(speech_timestamps):
                    start = ts["start"]
                    end = ts["end"]
                    chunk = wav[start:end].unsqueeze(0)
                    duration = (end - start) / 16000.0

                    chunk_name = f"{dialect_info['dialect'].lower()}_{speaker_tag}_{idx:04d}.wav"
                    chunk_path = os.path.join(PROCESSED_DIR, chunk_name)

                    torchaudio.save(chunk_path, chunk, 16000, encoding="PCM_S", bits_per_sample=16)

                    new_metadata_records.append({
                        "file_name": chunk_name,
                        "duration_sec": round(duration, 2),
                        "dialect": dialect_info["dialect"],
                        "region": dialect_info["region"],
                        "speaker_id": speaker_tag,
                        "source_title": re.sub(r"[^\w\s-]", "", title)[:50],
                        "source_url": f"https://www.youtube.com/watch?v={video_id}"
                    })
                    saved_chunks += 1

                existing_video_ids.add(video_id)
                print(f"   -> Extracted {saved_chunks} speech clips.")

            except Exception as err:
                print(f"Error chunking {video_id}: {err}")
            finally:
                if os.path.exists(raw_wav_path):
                    os.remove(raw_wav_path)

# -------------------------------------------------------------
# 5. RUN HARVESTER
# -------------------------------------------------------------
for item in TALUKA_SEARCH_CONFIG:
    search_and_download(item["query"], item["count"], item)

# -------------------------------------------------------------
# 6. APPEND & SAVE MASTER METADATA
# -------------------------------------------------------------
if new_metadata_records:
    new_df = pd.DataFrame(new_metadata_records)
    if os.path.exists(metadata_csv_path):
        existing_df = pd.read_csv(metadata_csv_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["file_name"])
        combined_df.to_csv(metadata_csv_path, index=False)
        print("\n" + "=" * 50)
        print(f"Appended {len(new_df)} new speech clips.")
        print(f"Master dataset now contains {len(combined_df)} total speech clips.")
        print("=" * 50)
    else:
        new_df.to_csv(metadata_csv_path, index=False)
        print(f"Created metadata with {len(new_df)} clips.")
else:
    print("\nNo new clips were extracted.")