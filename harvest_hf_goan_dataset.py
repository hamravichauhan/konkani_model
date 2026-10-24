#!/usr/bin/env python3
"""
Automated Goan Speech Harvester & Dialect Labeler
Dataset Source: Reubencf/goan-konkani-speech (Hugging Face)
Extracts, labels, downloads, converts to 16kHz WAV, and packages
fresh balanced training data for the 8 target Goan dialects.
"""

import os
import sys
import json
import re
import argparse
import tempfile
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

HF_BASE_URL = "https://huggingface.co/datasets/Reubencf/goan-konkani-speech/resolve/main"

# Dialect Keyword Map for Goan Talukas, Towns, Cultural and Dialectal markers
DIALECT_KEYWORDS = {
    'Bardez': {
        'region': 'North',
        'keywords': [
            'bardez', 'mapusa', 'mhapsa', 'mapxem', 'calangute', 'kalanvgot', 'candolim', 
            'anjuna', 'vagator', 'siolim', 'shivoli', 'aldona', 'porvorim', 'saligao', 
            'bastora', 'assagao', 'parra', 'nagoa', 'guirim', 'pilerne', 'sinquerim', 
            'nerul', 'tivim', 'colvale', 'moira', 'baga', 'arpora', 'sangolda', 'socorro', 
            'penha de franca', 'reismagos', 'reis magos', 'oxel', 'camurlim'
        ]
    },
    'Pednekar': {
        'region': 'North',
        'keywords': [
            'pernem', 'pedne', 'pednem', 'arambol', 'mandrem', 'morjim', 'querim', 
            'keri pedne', 'tiracol', 'torxem', 'chandel', 'tuem', 'agarwada', 
            'dhargal', 'virnoda', 'mopa', 'harmal', 'korgaon', 'korgao', 'paliem', 
            'alorna', 'ibrampur', 'tamboxem', 'casarvarnem', 'varconda'
        ]
    },
    'Canacona': {
        'region': 'South',
        'keywords': [
            'canacona', 'kankon', 'chaudi', 'agonda', 'palolem', 'poinguinim', 
            'loliem', 'galgibaga', 'sadolxem', 'cotigao', 'gaondongrem', 
            'partagal', 'khola', 'maxem', 'polem', 'talpona', 'shristhal', 
            'patnem', 'mashem', 'shristal', 'kola', 'cabo de rama', 'mallikarjun'
        ]
    },
    'Sattarkar': {
        'region': 'NorthEast',
        'keywords': [
            'sattari', 'satari', 'valpoi', 'walpoi', 'pissurlem', 'honda', 
            'thane', 'nagargao', 'guleli', 'birondem', 'mhadei', 'bhironda', 
            'surla', 'morlem', 'sancordem', 'paikul', 'amthane', 'massordem', 
            'kumthol', 'zarme', 'brahmakarmali', 'hedus', 'dhave'
        ]
    },
    'Antruz': {
        'region': 'Central',
        'keywords': [
            'antruz', 'ponda', 'fonda', 'bandora', 'bandivade', 'priol', 'marcel', 
            'marcaim', 'kavlem', 'mangueshi', 'mardol', 'veling', 'shiroda', 'borim', 
            'usgao', 'panaji', 'panjim', 'ponnje', 'ribandar', 'old goa', 'taleigao', 
            'caranzalem', 'bambolim', 'corlim', 'chorao', 'divar', 'kumbharjua', 
            'tiisk', 'curti', 'durbhat', 'ramnathi', 'shantadurga'
        ]
    },
    'Sashti_Christian': {
        'region': 'South',
        'keywords': [
            'sashti', 'salcete', 'salcet', 'nuvem', 'benaulim', 'curtorim', 'raia', 
            'loutolim', 'navelim', 'chinchinim', 'velim', 'assolna', 'betalbatim', 
            'colva', 'varca', 'cavelossim', 'majorda', 'utorda', 'cansaulim', 'verna', 
            'tiatr', 'tiatrist', 'igroz', 'padri', 'saibin', 'firgoz', 'noven', 
            'festa misa', 'konvent', 'katorze', 'sant anton', 'se cathedral', 'bom jesus'
        ]
    },
    'Sashti_Hindu': {
        'region': 'South',
        'keywords': [
            'madgaon', 'margao', 'modgonv', 'comba', 'aquem', 'damodar', 'davorlim', 
            'zambaulim', 'fatorpa', 'kunkalikarin', 'kuncolkar', 'fatorda', 'pajifond', 
            'borda', 'gogol', 'chandor', 'paroda', 'cuncolim'
        ]
    },
    'Gawda_Kunbi': {
        'region': 'Tribal',
        'keywords': [
            'gawda', 'gauda', 'kunbi', 'dhangar', 'velip', 'fugdi', 'dhalo', 'zaghor', 
            'ghumat', 'intruz', 'adivasi', 'sheti', 'kulagar', 'xetkar', 'xetkamti', 
            'dhonshi', 'shigmo', 'romtamel', 'talgadi', 'gopha', 'zula', 'folk dance'
        ]
    }
}

def classify_text(text):
    """Matches text against Goan regional & dialect markers."""
    clean_text = text.lower()
    for dialect, data in DIALECT_KEYWORDS.items():
        pattern = r'\b(' + '|'.join(data['keywords']) + r')\b'
        if re.search(pattern, clean_text):
            return dialect, data['region']
    return None, None

def download_and_process_clip(item, out_dir):
    """
    Downloads a single clip from Hugging Face and converts it to 16kHz mono WAV.
    """
    file_rel_path = item['file_name']
    url = f"{HF_BASE_URL}/{file_rel_path}"
    base_name = os.path.splitext(os.path.basename(file_rel_path))[0]
    dialect = item['dialect']
    out_wav_name = f"hf_{dialect.lower()}_{base_name}.wav"
    out_wav_path = os.path.join(out_dir, out_wav_name)

    if os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 1024:
        return {
            'file_name': out_wav_name,
            'duration_sec': item.get('duration', 0.0),
            'dialect': dialect,
            'region': item['region'],
            'speaker_id': item.get('speaker_gender', 'unknown'),
            'source_title': f"{item.get('channel', 'Goan TV')} - {item.get('video_id', '')}",
            'source_url': item.get('source_url', ''),
            'transcript': item.get('text', '')
        }

    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as tmp:
        tmp_flac = tmp.name

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            with open(tmp_flac, 'wb') as f:
                f.write(resp.read())

        # FFmpeg: Convert to 16kHz, mono, PCM 16-bit
        cmd = [
            'ffmpeg', '-y', '-i', tmp_flac,
            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
            out_wav_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        if os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 1024:
            return {
                'file_name': out_wav_name,
                'duration_sec': round(float(item.get('duration', 0.0)), 2),
                'dialect': dialect,
                'region': item['region'],
                'speaker_id': item.get('speaker_gender', 'unknown'),
                'source_title': f"{item.get('channel', 'Goan TV')} - {item.get('video_id', '')}",
                'source_url': item.get('source_url', ''),
                'transcript': item.get('text', '')
            }
    except Exception as e:
        # Ignore failed downloads/network hiccups
        pass
    finally:
        if os.path.exists(tmp_flac):
            try:
                os.remove(tmp_flac)
            except:
                pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Harvest and label free Goan Konkani speech clips")
    parser.add_argument("--metadata_file", default="dataset/raw_hf_metadata.jsonl", help="Path to raw metadata jsonl")
    parser.add_argument("--out_dir", default="dataset/processed_wavs", help="Where to save converted 16kHz wavs")
    parser.add_argument("--max_per_dialect", type=int, default=200, help="Target clips per dialect (default: 200)")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent download workers")
    parser.add_argument("--min_duration", type=float, default=3.0, help="Min clip length in seconds")
    parser.add_argument("--max_duration", type=float, default=20.0, help="Max clip length in seconds")
    parser.add_argument("--merge", action="store_true", help="Merge with existing metadata_clean.csv and regenerate splits")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.metadata_file):
        print(f"❌ Metadata file {args.metadata_file} not found! Downloading from Hugging Face...")
        url = "https://huggingface.co/datasets/Reubencf/goan-konkani-speech/resolve/main/metadata.jsonl"
        urllib.request.urlretrieve(url, args.metadata_file)
        print(f"✅ Downloaded {args.metadata_file}")

    print("🔍 Reading and classifying Goan speech transcripts...")
    dialect_buckets = defaultdict(list)
    with open(args.metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                dur = float(item.get('duration', 0.0))
                if dur < args.min_duration or dur > args.max_duration:
                    continue
                dialect, region = classify_text(item.get('text', ''))
                if dialect:
                    item['dialect'] = dialect
                    item['region'] = region
                    dialect_buckets[dialect].append(item)
            except:
                continue

    print("\n📊 Dialect matches discovered in Hugging Face corpus:")
    selected_clips = []
    for dialect, items in dialect_buckets.items():
        take_n = min(len(items), args.max_per_dialect)
        print(f"  • {dialect:20s}: {len(items):5d} total matches -> Selected {take_n}")
        selected_clips.extend(items[:take_n])

    print(f"\n🚀 Total clips queued for download & conversion: {len(selected_clips)}")
    print(f"⚡ Downloading and processing via {args.workers} parallel workers...")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_and_process_clip, item, args.out_dir): item for item in selected_clips}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Harvesting Audio"):
            res = future.result()
            if res is not None:
                results.append(res)

    print(f"\n✅ Successfully harvested & converted {len(results)} audio clips to 16kHz mono WAV!")

    # Save harvested metadata
    harvested_df = pd.DataFrame(results)
    harvested_csv = "dataset/metadata_hf_harvested.csv"
    harvested_df.to_csv(harvested_csv, index=False)
    print(f"💾 Saved harvested metadata to: {harvested_csv}")
    print("\nHarvested Breakdown:")
    print(harvested_df['dialect'].value_counts())

    # Optional: Merge with existing clean dataset
    if args.merge:
        existing_csv = "dataset/metadata_clean.csv"
        if os.path.exists(existing_csv):
            print(f"\n🔄 Merging with existing dataset: {existing_csv}...")
            base_df = pd.read_csv(existing_csv)
            # Align columns
            cols = ['file_name', 'duration_sec', 'dialect', 'region', 'speaker_id', 'source_title', 'source_url']
            merged_df = pd.concat([base_df[cols], harvested_df[cols]], ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=['file_name'])

            merged_csv = "dataset/metadata_clean_v6.csv"
            merged_df.to_csv(merged_csv, index=False)
            print(f"🎉 Merged Master Dataset saved: {merged_csv} ({len(merged_df)} total clips)")
            print("\nMaster v6 Dialect Distribution:")
            print(merged_df['dialect'].value_counts())

            # Generate stratified train/val splits (85% train, 15% val)
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                merged_df,
                test_size=0.15,
                random_state=42,
                stratify=merged_df['dialect']
            )
            train_df.to_csv("dataset/train_v6.csv", index=False)
            val_df.to_csv("dataset/val_v6.csv", index=False)
            print(f"📊 Created Stratified Splits: Train ({len(train_df)}) | Val ({len(val_df)})")

if __name__ == "__main__":
    main()
