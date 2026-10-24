import os
import sys
import tempfile
import subprocess
import torch
import torchaudio
import numpy as np
from transformers import AutoConfig, AutoModel
import torch.nn as nn

MODEL_ID = "facebook/mms-300m"
SAMPLE_RATE = 16000
MAX_AUDIO_SEC = 6.0

CHECKPOINT_PATH = "best_konkani_dialect_model_v5.pt"
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = "best_konkani_dialect_model.pt"
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = "checkpoints/best_konkani_dialect_model.pt"

DIALECT_MAP = {
    "Antruz": 0, "Bardez": 1, "Canacona": 2, "Gawda_Kunbi": 3,
    "Pednekar": 4, "Sashti_Christian": 5, "Sashti_Hindu": 6, "Sattarkar": 7
}
ID2DIALECT = {v: k for k, v in DIALECT_MAP.items()}

REGION_DETAILS = {
    "Antruz": ("Central Goa (Ponda / Priol)", "Standard Goan Konkani dialect, commonly spoken across Ponda, Priol, and formal Goan media."),
    "Bardez": ("North Goa (Bardez / Mapusa / Aldona / Siolim)", "Prominent northern dialect characterized by distinct open vowel endings and northern Goan cadence."),
    "Canacona": ("Far South Goa (Canacona taluka)", "Southern frontier dialect influenced by coastal Karwar phonology with unique vocabulary."),
    "Gawda_Kunbi": ("South Interior / Tribal Goa (Quepem / Sanguem / Canacona)", "Indigenous Goan dialect preserved by traditional Gawda and Kunbi communities with rich folk oral heritage."),
    "Pednekar": ("Far North Goa (Pernem taluka)", "Northern border dialect transitioning towards Malvani/Sindhudurg phonetics."),
    "Sashti_Christian": ("South Goa (Salcete Catholic / Margao / Benaulim)", "Salcete Catholic dialect, heavily featured in Goan Tiatr with classic Portuguese linguistic adaptations."),
    "Sashti_Hindu": ("South Goa (Salcete Hindu / Margao / Fatorda)", "Salcete Hindu dialect spoken predominantly by the Saraswat and resident Hindu communities of Margao."),
    "Sattarkar": ("North East Goa (Sattari / Valpoi / Ghat foothills)", "Ghat borderland dialect spoken in the Sahyadri foothills of Sattari.")
}

class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask=None):
        scores = self.attention(x)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1).bool(), -1e4)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1)

class MaskedAttentiveStatisticalPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask=None):
        scores = self.attention(x)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1).bool(), -1e4)
        weights = torch.softmax(scores, dim=1)
        mean = torch.sum(x * weights, dim=1)
        variance = torch.sum(weights * (x - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(torch.clamp(variance, min=1e-5))
        return torch.cat([mean, std], dim=-1)

class KonkaniDialectClassifier(nn.Module):
    def __init__(self, config, num_dialects=8, num_regions=7, is_statistical=True, has_region_classifier=False):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        self.is_statistical = is_statistical
        self.has_region_classifier = has_region_classifier
        hidden_dim = config.hidden_size
        pooled_dim = hidden_dim * 2 if is_statistical else hidden_dim

        self.pooler = MaskedAttentiveStatisticalPooling(hidden_dim) if is_statistical else AttentivePooling(hidden_dim)

        self.dialect_classifier = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_dialects)
        )

        if has_region_classifier:
            self.region_classifier = nn.Sequential(
                nn.Linear(pooled_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_regions)
            )

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        feat_mask = None
        if attention_mask is not None:
            feat_mask = self.encoder._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        pooled = self.pooler(hidden, mask=feat_mask)
        d_out = self.dialect_classifier(pooled)
        r_out = self.region_classifier(pooled) if self.has_region_classifier else None
        return d_out, r_out

def safe_convert_to_wav(audio_input):
    """
    4-Stage Studio Audio Preprocessing:
    1. Converts browser formats (webm, mp4, ogg, wav) to 16kHz mono PCM.
    2. High-pass filter (80 Hz) to eliminate table thumps, AC rumble, desk hum.
    3. Low-pass filter (7600 Hz) to cut out high-frequency digital noise and hiss.
    4. Adaptive FFT Denoising (afftdn) to cancel stationary fan/room noise.
    5. Silence Trimming (VAD): Trims leading dead air.
    """
    if not audio_input:
        return None
    out_wav = tempfile.mktemp(suffix=".wav")

    filter_chain = (
        "highpass=f=80,"
        "lowpass=f=7600,"
        "afftdn=nr=10:nf=-25,"
        "silenceremove=start_periods=1:start_duration=0.08:start_threshold=-40dB"
    )

    cmd = [
        "ffmpeg", "-y", "-i", str(audio_input),
        "-af", filter_chain,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
        out_wav
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 1024:
            return out_wav
        else:
            fallback_cmd = [
                "ffmpeg", "-y", "-i", str(audio_input),
                "-af", "highpass=f=80,lowpass=f=7600",
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
                out_wav
            ]
            subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return out_wav
    except Exception as e:
        print("FFmpeg conversion fallback:", e)
        cmd_basic = [
            "ffmpeg", "-y", "-i", str(audio_input),
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
            out_wav
        ]
        subprocess.run(cmd_basic, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return out_wav

def predict(audio_path):
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}.")
        return

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}.")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    wav_clean_path = safe_convert_to_wav(audio_path)
    try:
        waveform, sr = torchaudio.load(wav_clean_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
    finally:
        if wav_clean_path != audio_path and os.path.exists(wav_clean_path):
            os.remove(wav_clean_path)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    waveform = waveform.squeeze(0)
    max_len = int(MAX_AUDIO_SEC * SAMPLE_RATE)
    if len(waveform) > max_len:
        waveform = waveform[:max_len]
        mask = torch.ones(max_len, dtype=torch.long)
    else:
        actual_len = len(waveform)
        pad = max_len - actual_len
        waveform = torch.nn.functional.pad(waveform, (0, pad))
        mask = torch.cat([torch.ones(actual_len, dtype=torch.long), torch.zeros(pad, dtype=torch.long)])

    std = waveform.std()
    if std > 1e-5:
        waveform = (waveform - waveform.mean()) / std

    inputs = waveform.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    print(f"Loading checkpoint {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    if "projector.weight" in state_dict or "layer_weights" in state_dict:
        from transformers import Wav2Vec2ForSequenceClassification
        use_w = "layer_weights" in state_dict
        config = AutoConfig.from_pretrained(MODEL_ID, num_labels=len(DIALECT_MAP), use_weighted_layer_sum=use_w, classifier_proj_size=256)
        model = Wav2Vec2ForSequenceClassification(config).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        with torch.no_grad():
            outputs = model(input_values=inputs, attention_mask=mask)
            d_probs = torch.softmax(outputs.logits, dim=-1)[0]
    else:
        config = AutoConfig.from_pretrained(MODEL_ID)
        in_features = state_dict["dialect_classifier.0.weight"].shape[1]
        is_statistical = (in_features == 2048)
        has_region = any("region_classifier" in k for k in state_dict.keys())
        model = KonkaniDialectClassifier(config, is_statistical=is_statistical, has_region_classifier=has_region).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        with torch.no_grad():
            d_logits, _ = model(inputs, mask)
            d_probs = torch.softmax(d_logits, dim=-1)[0]

    top_d_idx = int(torch.argmax(d_probs).item())
    pred_dialect = ID2DIALECT[top_d_idx]
    pred_prob = float(d_probs[top_d_idx]) * 100

    region_name, context = REGION_DETAILS.get(pred_dialect, ("Unknown", "No description available."))

    print("\n" + "=" * 55)
    print("           KONKANI DIALECT PREDICTION")
    print("=" * 55)
    print(f"Predicted Dialect: {pred_dialect.upper()} ({pred_prob:.2f}% confidence)")
    print(f"Geographic Region: {region_name}")
    print(f"Context:           {context}")
    print("-" * 55)
    print("Top Probabilities:")
    top_k = torch.topk(d_probs, k=min(4, len(DIALECT_MAP)))
    for prob, idx in zip(top_k.values, top_k.indices):
        d_name = ID2DIALECT[int(idx.item())]
        print(f"  - {d_name:18s}: {prob.item() * 100:6.2f}%")
    print("=" * 55)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file>")
        sys.exit(1)
    predict(sys.argv[1])
