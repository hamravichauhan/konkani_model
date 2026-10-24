import os
import re
import time
import pandas as pd
import torch
import torchaudio
from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func

# -------------------------------------------------------------
# 1. CURATED HARVEST TARGETS (Focusing on under-represented classes)
# -------------------------------------------------------------
TARGET_VIDEOS = [
    # --- SASHTI HINDU (South Goa: Margao, Fatorda, Cuncolim) ---
    {
        "dialect": "Sashti_Hindu",
        "region": "South",
        "video_id": "VNBqgnOq40A",
        "title": "Prudent Media Matadar Raja Margao Fatorda"
    },
    {
        "dialect": "Sashti_Hindu",
        "region": "South",
        "video_id": "xuTqxD0MD4c",
        "title": "Nagar Goem Cuncolim Epi 09"
    },
    {
        "dialect": "Sashti_Hindu",
        "region": "South",
        "video_id": "6AUJYlNjXJk",
        "title": "Power crisis Margao IDC Ground Report"
    },
    {
        "dialect": "Sashti_Hindu",
        "region": "South",
        "video_id": "Wixt-nWEfI4",
        "title": "Ravindra Bhavan Margao Press Conference"
    },
    {
        "dialect": "Sashti_Hindu",
        "region": "South",
        "video_id": "IC7pgiGyunk",
        "title": "Cuncolim municipality locals demand"
    },
    {
        "dialect": "Sashti_Hindu",
        "region": "South",
        "video_id": "ZzNvTbDSsS0",
        "title": "Matadar Raja Cuncolim Velim Banulim"
    },

    # --- CANACONA (Far South Goa: Canacona, Poinguinim, Gaodongrem) ---
    {
        "dialect": "Canacona",
        "region": "Far_South",
        "video_id": "wtfIsGU8ikQ",
        "title": "Matadar Raja Canacona Prudent"
    },
    {
        "dialect": "Canacona",
        "region": "Far_South",
        "video_id": "Rzwf6u8RNs8",
        "title": "Matadar Raja Canacona Part 1"
    },
    {
        "dialect": "Canacona",
        "region": "Far_South",
        "video_id": "Ml_NyC02iy0",
        "title": "Exclusive hunt mood of people of Canacona"
    },
    {
        "dialect": "Canacona",
        "region": "Far_South",
        "video_id": "1vTeu60N6JA",
        "title": "Why Ravindra Bhavan Canacona Flooded"
    },
    {
        "dialect": "Canacona",
        "region": "Far_South",
        "video_id": "CWr6EF3wE70",
        "title": "Matadar Raja Canacona Taluka"
    },

    # --- PEDNEKAR (Far North Goa: Pernem, Mandrem, Morjim) ---
    {
        "dialect": "Pednekar",
        "region": "Far_North",
        "video_id": "tMfW-RT6kB4",
        "title": "Matadar Raja Episode 1 Mandrem"
    },
    {
        "dialect": "Pednekar",
        "region": "Far_North",
        "video_id": "R_0psjfey1I",
        "title": "Mandrem Constituency Matadar Raja Ladies Special"
    },
    {
        "dialect": "Pednekar",
        "region": "Far_North",
        "video_id": "sA9FuoRXFzQ",
        "title": "Illegal sand mining in Pernem locals speaking"
    },
    {
        "dialect": "Pednekar",
        "region": "Far_North",
        "video_id": "hx2TCwssvLY",
        "title": "Pernem Bahujan Samaj demand compensation"
    },
    {
        "dialect": "Pednekar",
        "region": "Far_North",
        "video_id": "gQ5Z-ARoMcQ",
        "title": "Pernem residents agitated with road digging"
    },

    # --- BARDEZ (North Goa: Mapusa, Siolim, Aldona, Saligao) ---
    {
        "dialect": "Bardez",
        "region": "North",
        "video_id": "DzmP-_Mh6HU",
        "title": "Matadar Raja Mapusa"
    },
    {
        "dialect": "Bardez",
        "region": "North",
        "video_id": "HXket-01Ny0",
        "title": "Matadar Raja Episode 3 Siolim"
    },
    {
        "dialect": "Bardez",
        "region": "North",
        "video_id": "Kuc_eyMIRp8",
        "title": "Matadar Raja Saligao Calangute Siolim"
    },
    {
        "dialect": "Bardez",
        "region": "North",
        "video_id": "wWV8-Xx0S18",
        "title": "TMC Mapusa Candidate interview"
    },
    {
        "dialect": "Bardez",
        "region": "North",
        "video_id": "FnBrO07MPJo",
        "title": "MLA Joshua meets Mapusa market vendors"
    },

    # --- SATTARKAR (North East Goa: Valpoi, Sattari, Poriem) ---
    {
        "dialect": "Sattarkar",
        "region": "North_East",
        "video_id": "Vyq2hKws8sg",
        "title": "Matadar Raja Valpoi"
    },
    {
        "dialect": "Sattarkar",
        "region": "North_East",
        "video_id": "Cy5u8uOUaRI",
        "title": "Nagar Goem Valpoi Epi 06"
    },
    {
        "dialect": "Sattarkar",
        "region": "North_East",
        "video_id": "PL2mQva80X4",
        "title": "Sattari forest clashes with locals"
    },
    {
        "dialect": "-jZA1RF2pOM",
        "dialect": "Sattarkar",
        "region": "North_East",
        "video_id": "-jZA1RF2pOM",
        "title": "Sattari locals protest harassment by foresters"
    },
    {
        "dialect": "Sattarkar",
        "region": "North_East",
        "video_id": "YWncH3nb8_8",
        "title": "Matadar Raja Valpoi and Parye constituency"
    },

    # --- GAWDA KUNBI (South Interior: Sanguem, Quepem, Dharbandora) ---
    {
        "dialect": "Gawda_Kunbi",
        "region": "South_Interior",
        "video_id": "8q5wnbAjBaI",
        "title": "Matadar Raja Quepem Sanguem Dharbandora"
    },
    {
        "dialect": "Gawda_Kunbi",
        "region": "South_Interior",
        "video_id": "G9Ui0uJtBbo",
        "title": "Prudent Media Matadar Raja Sanguem"
    },

    # --- ANTRUZ (Central Goa: Ponda, Madkai) ---
    {
        "dialect": "Antruz",
        "region": "Central",
        "video_id": "2bqBdMRT2Og",
        "title": "Prudent Media Matadar Raja Ponda"
    },
    {
        "dialect": "Antruz",
        "region": "Central",
        "video_id": "SVKQXBzNzUY",
        "title": "Matadar Raja Madkai"
    },

    # --- SASHTI CHRISTIAN (South Goa: Benaulim, Nuvem) ---
    {
        "dialect": "Sashti_Christian",
        "region": "South",
        "video_id": "2HzBhUQYkjQ",
        "title": "Matadar Raja Benaulim South Goa"
    }
]

RAW_TEMP = "dataset/raw_temp"
PROCESSED_DIR = "dataset/processed_wavs"
HARVEST_META_PATH = "dataset/metadata_harvested.csv"
MAX_CHUNKS_PER_VIDEO = 35

os.makedirs(RAW_TEMP, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -------------------------------------------------------------
# 2. CHECK ALREADY PROCESSED VIDEOS
# -------------------------------------------------------------
existing_vids = set()
for path in ["dataset/metadata.csv", "dataset/metadata_clean.csv", HARVEST_META_PATH]:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            for u in df["source_url"].dropna():
                if "watch?v=" in u:
                    existing_vids.add(u.split("watch?v=")[-1].strip())
        except Exception:
            pass

print(f"Skipping {len(existing_vids)} already downloaded videos.")

# -------------------------------------------------------------
# 3. LOAD SILERO VAD
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
# 4. DOWNLOAD & EXTRACT PIPELINE
# -------------------------------------------------------------
new_records = []
if os.path.exists(HARVEST_META_PATH):
    try:
        new_records = pd.read_csv(HARVEST_META_PATH).to_dict("records")
    except Exception:
        pass

total_videos = len(TARGET_VIDEOS)

for idx, item in enumerate(TARGET_VIDEOS, 1):
    vid = item["video_id"]
    dialect = item["dialect"]
    region = item["region"]
    title_hint = item["title"]

    if vid in existing_vids:
        print(f"[{idx}/{total_videos}] SKIP: {vid} ({dialect}) - already processed")
        continue

    url = f"https://www.youtube.com/watch?v={vid}"
    print(f"\n[{idx}/{total_videos}] Processing: {dialect} ({region}) -> {title_hint} ({vid})")

    out_base = os.path.join(RAW_TEMP, f"{vid}")
    out_wav = f"{out_base}.wav"

    ydl_opts = {
        "format": "ba/b[ext=m4a]/18/best",
        "outtmpl": f"{out_base}.%(ext)s",
        "download_ranges": download_range_func(None, [(45, 645)]),  # 10 minutes (0:45 to 10:45)
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "postprocessor_args": [
            "-ar", "16000",
            "-ac", "1"
        ],
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios"]
            }
        },
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"   [Error downloading {vid}]: {e}")
        continue

    if not os.path.exists(out_wav):
        print(f"   [Warning] WAV file not produced for {vid}")
        continue

    # Perform VAD Chunking
    try:
        wav = read_audio(out_wav, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            sampling_rate=16000,
            min_speech_duration_ms=2000,
            max_speech_duration_s=7.0,
            min_silence_duration_ms=400
        )

        speaker_tag = f"spk_{vid[:6]}"
        chunks_saved = 0

        for s_idx, ts in enumerate(speech_timestamps):
            if chunks_saved >= MAX_CHUNKS_PER_VIDEO:
                break

            start = ts["start"]
            end = ts["end"]
            chunk = wav[start:end].unsqueeze(0)
            duration = (end - start) / 16000.0

            chunk_filename = f"{dialect.lower()}_{speaker_tag}_{chunks_saved:04d}.wav"
            chunk_filepath = os.path.join(PROCESSED_DIR, chunk_filename)

            torchaudio.save(chunk_filepath, chunk, 16000, encoding="PCM_S", bits_per_sample=16)

            record = {
                "file_name": chunk_filename,
                "duration_sec": round(duration, 2),
                "dialect": dialect,
                "region": region,
                "speaker_id": speaker_tag,
                "source_title": re.sub(r"[^\w\s-]", "", title_hint)[:50],
                "source_url": url
            }
            new_records.append(record)
            chunks_saved += 1

        existing_vids.add(vid)
        print(f"   -> Extracted {chunks_saved} clean speech chunks (speaker: {speaker_tag})")

        # Save progress after every video
        pd.DataFrame(new_records).to_csv(HARVEST_META_PATH, index=False)

    except Exception as err:
        print(f"   [Error during VAD chunking {vid}]: {err}")
    finally:
        if os.path.exists(out_wav):
            try:
                os.remove(out_wav)
            except Exception:
                pass

print("\n" + "=" * 55)
print(f"HARVEST COMPLETE: {len(new_records)} clips harvested.")
print(f"Saved to: {HARVEST_META_PATH}")
print("=" * 55)
