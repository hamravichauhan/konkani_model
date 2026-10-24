import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import pandas as pd
from transformers import AutoModel, AutoConfig
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

# -------------------------------------------------------------
# 1. HARDWARE & HYPERPARAMETERS
# -------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Executing on device: {DEVICE}")

MODEL_ID = "facebook/mms-300m"
BATCH_SIZE = 8 if DEVICE.type == "cuda" else 4
GRAD_ACCUM_STEPS = 2
EPOCHS = 15
MAX_AUDIO_SEC = 5.0
SAMPLE_RATE = 16000

DIALECT_MAP = {
    "Antruz": 0, "Bardez": 1, "Canacona": 2, "Gawda_Kunbi": 3,
    "Pednekar": 4, "Sashti_Christian": 5, "Sashti_Hindu": 6, "Sattarkar": 7
}
ID2DIALECT = {v: k for k, v in DIALECT_MAP.items()}

REGION_MAP = {
    "Central": 0, "North": 1, "Far_South": 2, "South_Interior": 3,
    "Far_North": 4, "South": 5, "North_East": 6
}
ID2REGION = {v: k for k, v in REGION_MAP.items()}

# -------------------------------------------------------------
# 2. DATASET & AUDIO AUGMENTATION
# -------------------------------------------------------------
class KonkaniDataset(Dataset):
    def __init__(self, csv_path, audio_dir, max_sec=5.0, augment=False):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.max_length = int(max_sec * SAMPLE_RATE)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row["file_name"])

        waveform, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        waveform = waveform.squeeze(0)

        # Random temporal crop during training
        if len(waveform) > self.max_length:
            if self.augment:
                max_start = len(waveform) - self.max_length
                start = torch.randint(0, max_start, (1,)).item()
                waveform = waveform[start:start + self.max_length]
            else:
                waveform = waveform[:self.max_length]
        else:
            padding = self.max_length - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Real-world microphone simulation augmentations
        if self.augment:
            # Volume jitter (0.8x to 1.2x)
            waveform = waveform * torch.empty(1).uniform_(0.8, 1.2).item()

            # Ambient noise injection (20dB to 32dB SNR)
            if torch.rand(1).item() > 0.4:
                noise = torch.randn_like(waveform)
                snr = 10 ** (torch.randint(20, 32, (1,)).item() / 20.0)
                scale = waveform.norm(2) / (snr * max(noise.norm(2).item(), 1e-4))
                waveform = waveform + scale * noise

            # Time masking
            if torch.rand(1).item() > 0.5:
                mask_len = torch.randint(1600, 6400, (1,)).item()
                mask_start = torch.randint(0, max(1, len(waveform) - mask_len), (1,)).item()
                waveform[mask_start:mask_start + mask_len] = 0.0

        return {
            "input_values": waveform,
            "dialect_label": torch.tensor(DIALECT_MAP[row["dialect"]], dtype=torch.long),
            "region_label": torch.tensor(REGION_MAP[row["region"]], dtype=torch.long)
        }

def collate_fn(batch):
    return {
        "input_values": torch.stack([b["input_values"] for b in batch]),
        "dialect_label": torch.stack([b["dialect_label"] for b in batch]),
        "region_label": torch.stack([b["region_label"] for b in batch])
    }

# -------------------------------------------------------------
# 3. ARCHITECTURE WITH SELECTIVE LAYER FINE-TUNING
# -------------------------------------------------------------
class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1)

class KonkaniDialectClassifier(nn.Module):
    def __init__(self, model_id, num_dialects=8, num_regions=7, unfreeze_top_layers=4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        self.encoder.feature_extractor._freeze_parameters()

        # Freeze all lower layers
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Selectively unfreeze top transformer layers
        num_layers = len(self.encoder.encoder.layers)
        unfreeze_start = num_layers - unfreeze_top_layers
        for i in range(unfreeze_start, num_layers):
            for param in self.encoder.encoder.layers[i].parameters():
                param.requires_grad = True

        hidden_dim = self.encoder.config.hidden_size
        self.pooler = AttentivePooling(hidden_dim)

        self.dialect_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_dialects)
        )

        self.region_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_regions)
        )

    def forward(self, input_values):
        outputs = self.encoder(input_values)
        pooled = self.pooler(outputs.last_hidden_state)
        return self.dialect_classifier(pooled), self.region_classifier(pooled)

# -------------------------------------------------------------
# 4. TRAINING & EVALUATION
# -------------------------------------------------------------
def main():
    audio_dir = "dataset/processed_wavs"
    train_df = pd.read_csv("dataset/train.csv")
    val_df = pd.read_csv("dataset/val.csv")

    train_ds = KonkaniDataset("dataset/train.csv", audio_dir, MAX_AUDIO_SEC, augment=True)
    val_ds = KonkaniDataset("dataset/val.csv", audio_dir, MAX_AUDIO_SEC, augment=False)

    # Balanced Sampler to eliminate class imbalance
    counts = train_df["dialect"].value_counts()
    weights_map = {d: 1.0 / count for d, count in counts.items()}
    sample_weights = [weights_map[d] for d in train_df["dialect"]]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_df), replacement=True)

    num_workers = 2 if DEVICE.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    print(f"\nModel: {MODEL_ID}")
    print(f"Training on: {len(train_ds)} clips | Validation on: {len(val_ds)} clips")
    model = KonkaniDialectClassifier(MODEL_ID, num_dialects=len(DIALECT_MAP), num_regions=len(REGION_MAP), unfreeze_top_layers=4).to(DEVICE)

    # Differential Learning Rates
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.pooler.parameters()) + list(model.dialect_classifier.parameters()) + list(model.region_classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1.5e-5},
        {"params": head_params, "lr": 3e-4}
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion_dialect = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_region = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_f1 = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d} [Train]")
        for step, batch in enumerate(loop):
            inputs = batch["input_values"].to(DEVICE)
            d_labels = batch["dialect_label"].to(DEVICE)
            r_labels = batch["region_label"].to(DEVICE)

            d_preds, r_preds = model(inputs)
            loss_d = criterion_dialect(d_preds, d_labels)
            loss_r = criterion_region(r_preds, r_labels)
            loss = (loss_d + 0.3 * loss_r) / GRAD_ACCUM_STEPS

            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * GRAD_ACCUM_STEPS
            loop.set_postfix(loss=loss.item() * GRAD_ACCUM_STEPS)

            if step % 50 == 0 and DEVICE.type == "mps":
                torch.mps.empty_cache()

        scheduler.step()
        avg_train = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        all_d_preds, all_d_targets = [], []

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d} [Val]")
        with torch.no_grad():
            for batch in val_loop:
                inputs = batch["input_values"].to(DEVICE)
                d_labels = batch["dialect_label"].to(DEVICE)
                r_labels = batch["region_label"].to(DEVICE)

                d_preds, r_preds = model(inputs)
                loss_d = criterion_dialect(d_preds, d_labels)
                loss_r = criterion_region(r_preds, r_labels)
                val_loss = loss_d + 0.3 * loss_r

                total_val_loss += val_loss.item()
                all_d_preds.extend(torch.argmax(d_preds, dim=-1).cpu().numpy())
                all_d_targets.extend(d_labels.cpu().numpy())

        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        avg_val = total_val_loss / len(val_loader)
        val_acc = (np.array(all_d_preds) == np.array(all_d_targets)).mean() * 100
        val_f1 = f1_score(all_d_targets, all_d_preds, average="macro") * 100

        print(f"\nResult -> Epoch {epoch:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {val_acc:.2f}% | Val Macro F1: {val_f1:.2f}%\n")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "dialect_map": DIALECT_MAP,
                "region_map": REGION_MAP,
                "epoch": epoch,
                "val_macro_f1": val_f1,
                "val_acc": val_acc
            }, "checkpoints/best_konkani_dialect_model.pt")
            print(f"  --> Saved new best checkpoint (Macro F1: {val_f1:.2f}%) to checkpoints/best_konkani_dialect_model.pt\n")

    print("\nTraining completed.")
    print(classification_report(all_d_targets, all_d_preds, target_names=[ID2DIALECT[i] for i in range(len(DIALECT_MAP))]))

if __name__ == "__main__":
    main()
