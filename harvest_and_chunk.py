import os
import torch
import torchaudio
import pandas as pd
from yt_dlp import YoutubeDL

# Define target sources mapped to dialects/regions
SOURCE_MANIFEST = [
    {
        "url": "https://www.youtube.com/watch?v=YOUR_ANTRUZ_VIDEO_ID",
        "dialect": "Antruz",
        "region": "Central",
        "speaker_id": "spk_pond_01"
    },
    {
        "url": "https://www.youtube.com/watch?v=YOUR_SAXTTI_VIDEO_ID",
        "dialect": "Saxtti",
        "region": "South",
        "speaker_id": "spk_salc_01"
    },
    {
        "url": "https://www.youtube.com/watch?v=YOUR_BARDEZ_VIDEO_ID",
        "dialect": "Bardez",
        "region": "North",
        "speaker_id": "spk_bard_01"
    },
    # 1. Antruz (Central / Ponda / Standard)
    # Long news discussions & debates have clear, un-accented Standard/Antruz speech
    {
        "url": "https://www.youtube.com/watch?v=HT5Z6V3Ycmw",  # Tiatr / Stage audio
        "dialect": "Saxtti",
        "region": "South",
        "speaker_id": "spk_tiatr_south_01"
    },
    {
        # Prudent Media Goa discussion (Antruz/Standard news panel)
        "url": "https://www.youtube.com/watch?v=PcllrWhcba4",
        "dialect": "Antruz",
        "region": "Central",
        "speaker_id": "spk_air_central_01"
    },
    # 2. Saxtti (South / Salcete)
    {
        # Full Goan Tiatr dialogue scenes (Heavy South Salcete accent)
        "url": "https://www.youtube.com/watch?v=IV0lvaqy8Ng",
        "dialect": "Saxtti",
        "region": "South",
        "speaker_id": "spk_tiatr_south_02"
    }
]

RAW_DIR = "dataset/raw_downloads"
PROCESSED_DIR = "dataset/processed_wavs"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 1. Download audio as 16kHz mono WAV using yt-dlp
def download_audio(url, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'postprocessor_args': [
            '-ar', '16000',
            '-ac', '1'
        ],
        'quiet': True
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# 2. Load Silero VAD for smart pause/voice splitting
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, _, read_audio, _, _) = utils

all_metadata = []

for idx, entry in enumerate(SOURCE_MANIFEST):
    print(f"\nProcessing {entry['dialect']} from {entry['url']}...")
    raw_wav = os.path.join(RAW_DIR, f"temp_{idx}.wav")
    
    try:
        download_audio(entry["url"], os.path.join(RAW_DIR, f"temp_{idx}"))
    except Exception as e:
        print(f"Download failed for {entry['url']}: {e}")
        continue

    # Read the audio (16kHz expected by Silero)
    wav = read_audio(raw_wav, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(
        wav, 
        vad_model, 
        sampling_rate=16000,
        min_speech_duration_ms=2500,  # Min ~2.5s
        max_speech_duration_s=8.0     # Max 8s chunks
    )

    # Slice and save chunks
    for s_idx, ts in enumerate(speech_timestamps):
        start_sample = ts['start']
        end_sample = ts['end']
        
        chunk = wav[start_sample:end_sample].unsqueeze(0)
        duration = (end_sample - start_sample) / 16000.0
        
        chunk_filename = f"{entry['dialect'].lower()}_{entry['speaker_id']}_{s_idx:04d}.wav"
        chunk_filepath = os.path.join(PROCESSED_DIR, chunk_filename)
        
        torchaudio.save(chunk_filepath, chunk, 16000, encoding="PCM_S", bits_per_sample=16)
        
        all_metadata.append({
            "file_name": chunk_filename,
            "duration_sec": round(duration, 2),
            "dialect": entry["dialect"],
            "region": entry["region"],
            "speaker_id": entry["speaker_id"]
        })

    # Clean up raw temp file
    if os.path.exists(raw_wav):
        os.remove(raw_wav)

# 3. Export Master Metadata
df = pd.DataFrame(all_metadata)
df.to_csv("dataset/metadata.csv", index=False)
print(f"\nCompleted! Generated {len(df)} speech chunks in dataset/processed_wavs.")