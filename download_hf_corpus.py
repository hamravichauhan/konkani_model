import os
import pandas as pd
import soundfile as sf
import torchaudio
from datasets import load_dataset

OUTPUT_DIR = "dataset/open_corpus"
AUDIO_OUT = os.path.join(OUTPUT_DIR, "wavs")
os.makedirs(AUDIO_OUT, exist_ok=True)

print("Fetching Goan Konkani dataset from Hugging Face...")
# Loads the public Goan Konkani speech corpus (or ai4bharat/indic-speech)
try:
    ds = load_dataset("Reubencf/goan-konkani-speech", split="train", streaming=True)
except Exception as e:
    print(f"Dataset load fallback: {e}")
    ds = None

metadata_rows = []
TARGET_COUNT = 500  # Adjust as needed

if ds:
    for idx, item in enumerate(ds):
        if idx >= TARGET_COUNT:
            break

        audio_data = item.get("audio", {})
        array = audio_data.get("array")
        orig_sr = audio_data.get("sampling_rate", 16000)

        if array is None:
            continue

        filename = f"open_konkani_{idx:05d}.wav"
        filepath = os.path.join(AUDIO_OUT, filename)

        # Convert to torch tensor for resampling if needed
        tensor_audio = torchaudio.transforms.Resample(orig_sr, 16000)(
            torch.tensor(array).float().unsqueeze(0)
        )

        torchaudio.save(filepath, tensor_audio, 16000, encoding="PCM_S", bits_per_sample=16)

        metadata_rows.append({
            "file_name": filename,
            "dialect": "Standard_Goan",
            "region": "Central",
            "source": item.get("channel", "OpenCorpus")
        })

    pd.DataFrame(metadata_rows).to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
    print(f"Saved {len(metadata_rows)} audio clips to {AUDIO_OUT}")