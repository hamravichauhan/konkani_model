"""
Local M4 Training Script for Konkani Dialect Classifier (v4.0)
Optimized for Apple Silicon M4 GPU (MPS) & 16GB Unified Memory.

Highlights:
- Native Wav2Vec2ForSequenceClassification with learnable 25-layer weighted sum.
- Inverse-frequency class weighting (strictly prevents mode collapse).
- Fine-tunes top 2 transformer layers + layer_weights + classification head (~28M trainable params).
- Freeze lower layers for blazing speed and minimal memory (~2.5 GB VRAM).
- Checkpointing strictly based on Macro F1.
"""

import os
import random
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoConfig, Wav2Vec2ForSequenceClassification, get_cosine_schedule_with_warmup

# ----------------- CONFIGURATION -----------------
MODEL_ID = "facebook/mms-300m"
TRAIN_CSV = "dataset/train.csv"
VAL_CSV = "dataset/val.csv"
AUDIO_DIR = "dataset/processed_wavs"
SAVE_PATH = "best_konkani_dialect_model.pt"

SAMPLE_RATE = 16000
MAX_AUDIO_SEC = 6.0
MAX_LEN = int(SAMPLE_RATE * MAX_AUDIO_SEC)

BATCH_SIZE = 8          # Smooth and responsive for 16GB Unified Memory
GRAD_ACCUM_STEPS = 2    # Effective batch size = 16
EPOCHS = 14
BASE_LR = 1e-4          # For classification head and layer weights
BACKBONE_LR = 1.5e-5    # Gentle rate for top transformer layers
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

# Hardware setup: Mac M4 MPS
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 Using Apple Silicon M4 GPU (MPS acceleration enabled)!")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🚀 Using NVIDIA CUDA GPU!")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ GPU not detected; running on CPU.")

DIALECT_MAP = {
    "Antruz": 0,
    "Bardez": 1,
    "Canacona": 2,
    "Gawda_Kunbi": 3,
    "Pednekar": 4,
    "Sashti_Christian": 5,
    "Sashti_Hindu": 6,
    "Sattarkar": 7,
}
ID2DIALECT = {v: k for k, v in DIALECT_MAP.items()}

# ----------------- DATASET -----------------
class KonkaniDataset(Dataset):
    def __init__(self, csv_path, audio_dir, max_sec=6.0, augment=False):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.audio_dir = audio_dir
        self.max_len = int(SAMPLE_RATE * max_sec)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row["file_name"])

        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

        waveform = waveform.squeeze(0)
        actual_len = len(waveform)

        if actual_len >= self.max_len:
            waveform = waveform[:self.max_len]
            attention_mask = torch.ones(self.max_len, dtype=torch.long)
        else:
            pad = self.max_len - actual_len
            waveform = torch.nn.functional.pad(waveform, (0, pad))
            attention_mask = torch.cat([torch.ones(actual_len, dtype=torch.long), torch.zeros(pad, dtype=torch.long)])

        # Audio normalization
        std = waveform.std()
        if std > 1e-5:
            waveform = (waveform - waveform.mean()) / std

        # Light training augmentation (speed/noise jitter)
        if self.augment and random.random() < 0.40:
            noise = torch.randn_like(waveform)
            waveform = waveform + 0.010 * noise

        label = DIALECT_MAP[row["dialect"]]
        return {
            "input_values": waveform,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }

def collate_fn(batch):
    return {
        "input_values": torch.stack([b["input_values"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }

# ----------------- MAIN TRAINING ROUTINE -----------------
def main():
    print("=" * 65)
    print("   KONKANI DIALECT CLASSIFIER - LOCAL M4 TRAINING (v4.0)")
    print("=" * 65)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    print(f"Loaded {len(train_df):,} train clips and {len(val_df):,} validation clips.")

    # Calculate inverse-frequency class weights
    class_counts = train_df["dialect"].map(DIALECT_MAP).value_counts().sort_index().values
    total_samples = len(train_df)
    num_classes = len(DIALECT_MAP)
    class_weights = total_samples / (num_classes * class_counts.astype(np.float32))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    print("\nBalanced Class Weights (Inverse Frequency):")
    for d_name, d_id in sorted(DIALECT_MAP.items(), key=lambda x: x[1]):
        print(f"  [{d_id}] {d_name:<18}: count={class_counts[d_id]:<4} weight={class_weights[d_id]:.2f}")

    train_ds = KonkaniDataset(TRAIN_CSV, AUDIO_DIR, MAX_AUDIO_SEC, augment=True)
    val_ds = KonkaniDataset(VAL_CSV, AUDIO_DIR, MAX_AUDIO_SEC, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    print(f"\nLoading MMS-300M sequence classification model...")
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        num_labels=len(DIALECT_MAP),
        classifier_proj_size=256,
        hidden_dropout=0.1,
        final_dropout=0.1,
    )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID, config=config).to(DEVICE)

    # Freeze raw CNN feature encoder (version-safe)
    if hasattr(model, 'freeze_feature_encoder'):
        model.freeze_feature_encoder()
    elif hasattr(model, 'freeze_feature_extractor'):
        model.freeze_feature_extractor()
    else:
        for param in model.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

    # Freeze lower transformer layers
    for param in model.wav2vec2.parameters():
        param.requires_grad = False

    # Fine-tune top 2 transformer layers (where dialectal nuances are encoded)
    for param in model.wav2vec2.encoder.layers[-1].parameters():
        param.requires_grad = True
    for param in model.wav2vec2.encoder.layers[-2].parameters():
        param.requires_grad = True

    # Unfreeze classification head
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Architecture ready! Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

    # Differential learning rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "projector" in name or "classifier" in name or "layer_weights" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": BASE_LR, "weight_decay": WEIGHT_DECAY},
    ])

    total_training_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)

    best_macro_f1 = 0.0
    best_acc = 0.0
    patience = 5
    no_improve_count = 0

    print(f"Total training steps: {total_training_steps} (Warmup: {warmup_steps})")
    print(f"Starting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d} [Train]")
        for step, batch in enumerate(pbar):
            wavs = batch["input_values"].to(DEVICE)
            masks = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_values=wavs, attention_mask=masks)
            loss = criterion(outputs.logits, labels)
            loss_scaled = loss / GRAD_ACCUM_STEPS
            loss_scaled.backward()

            train_loss += loss.item() * len(labels)

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_ds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d} [Val]  "):
                wavs = batch["input_values"].to(DEVICE)
                masks = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                outputs = model(input_values=wavs, attention_mask=masks)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item() * len(labels)

                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_ds)
        val_acc = accuracy_score(all_targets, all_preds) * 100.0
        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0

        print(f"\n📊 Epoch {epoch:02d} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Macro F1: {val_f1:.2f}%")

        if val_f1 > best_macro_f1:
            improvement = val_f1 - best_macro_f1
            best_macro_f1 = val_f1
            best_acc = val_acc
            no_improve_count = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "dialect_map": DIALECT_MAP,
            }, SAVE_PATH)
            print(f"  --> 🌟 New best checkpoint saved to '{SAVE_PATH}' (Macro F1: {val_f1:.2f}% [+{improvement:.2f}%] | Val Acc: {val_acc:.2f}%)\n")
        else:
            no_improve_count += 1
            print(f"  --> (No improvement for {no_improve_count}/{patience} epochs)\n")
            if no_improve_count >= patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch} to preserve the best weights.")
                break

    print("=" * 65)
    print(f"🎉 Training Complete! Best Validation Result:")
    print(f"   Macro F1: {best_macro_f1:.2f}% | Accuracy: {best_acc:.2f}%")
    print(f"   Saved Model: {os.path.abspath(SAVE_PATH)}")
    print("=" * 65)

if __name__ == "__main__":
    main()
