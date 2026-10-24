import os
import tempfile
import subprocess
import torch
import torchaudio
import numpy as np
import gradio as gr
from transformers import AutoConfig, AutoModel
import torch.nn as nn

MODEL_ID = "facebook/mms-300m"
SAMPLE_RATE = 16000
MAX_AUDIO_SEC = 6.0  # 6.0s full-sentence cadence window

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
    "Antruz": ("Central Goa (Ponda / Priol)", "Standard Goan Konkani dialect, commonly spoken across Ponda, Priol, and formal Goan media with balanced cadence."),
    "Bardez": ("North Goa (Bardez / Mapusa / Aldona / Siolim)", "Prominent northern dialect characterized by distinct open vowel endings and northern Goan cadence."),
    "Canacona": ("Far South Goa (Canacona taluka)", "Southern frontier dialect influenced by coastal Karwar phonology with unique regional vocabulary."),
    "Gawda_Kunbi": ("South Interior / Tribal Goa (Quepem / Sanguem / Canacona)", "Indigenous Goan dialect preserved by traditional Gawda and Kunbi communities with rich folk heritage."),
    "Pednekar": ("Far North Goa (Pernem taluka)", "Northern border dialect transitioning towards Malvani/Sindhudurg phonetics and fast clip cadence."),
    "Sashti_Christian": ("South Goa (Salcete Catholic / Margao / Benaulim)", "Salcete Catholic dialect, heavily featured in Goan Tiatr with classic Portuguese linguistic adaptations and elongated trailing vowels."),
    "Sashti_Hindu": ("South Goa (Salcete Hindu / Margao / Fatorda)", "Salcete Hindu dialect spoken predominantly by the resident Hindu communities of Margao, with crisp dental articulation."),
    "Sattarkar": ("North East Goa (Sattari / Valpoi / Ghat foothills)", "Ghat borderland dialect spoken in the Sahyadri foothills of Sattari, featuring transitional Marathi-Konkani inflections.")
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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ensemble_models = []

def load_model():
    """Returns the primary ensemble models."""
    global ensemble_models
    if ensemble_models:
        return ensemble_models

    candidates = [
        ("Model v7.0 Deep-8", "best_konkani_dialect_model_v7_deep.pt", 0.40),
        ("Model v6.0 Pro", "best_konkani_dialect_model_v6.pt", 0.35),
        ("Model v5.0 Pro", "best_konkani_dialect_model_v5.pt", 0.35),
        ("Model Base", "best_konkani_dialect_model.pt", 0.25),
    ]

    loaded = []
    seen_sizes = set()
    for name, path, weight in candidates:
        if os.path.exists(path):
            fsize = os.path.getsize(path)
            # Avoid loading duplicate copies of the same weights
            if fsize in seen_sizes:
                continue
            seen_sizes.add(fsize)
            try:
                print(f"Loading {name} from {path} on {device}...")
                ckpt = torch.load(path, map_location=device, weights_only=False)
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

                if "projector.weight" in state_dict or "layer_weights" in state_dict or "classifier.weight" in state_dict:
                    from transformers import Wav2Vec2ForSequenceClassification
                    use_w = "layer_weights" in state_dict
                    config = AutoConfig.from_pretrained(MODEL_ID, num_labels=len(DIALECT_MAP), use_weighted_layer_sum=use_w, classifier_proj_size=256)
                    m = Wav2Vec2ForSequenceClassification(config).to(device)
                    m.load_state_dict(state_dict, strict=True)
                    m.eval()
                    loaded.append((name, m, weight))
                    print(f"✅ {name} loaded successfully!")
                else:
                    config = AutoConfig.from_pretrained(MODEL_ID)
                    in_features = state_dict["dialect_classifier.0.weight"].shape[1]
                    is_statistical = (in_features == 2048)
                    has_region = any("region_classifier" in k for k in state_dict.keys())
                    m = KonkaniDialectClassifier(config, is_statistical=is_statistical, has_region_classifier=has_region).to(device)
                    m.load_state_dict(state_dict, strict=True)
                    m.eval()
                    loaded.append((name, m, weight))
                    print(f"✅ {name} (Custom Classifier) loaded successfully!")
            except Exception as e:
                print(f"⚠️ Error loading {name} from {path}: {e}")

    if loaded:
        tot_w = sum(w for _, _, w in loaded)
        ensemble_models = [(n, m, w / tot_w) for n, m, w in loaded]
        print(f"🌟 Ensemble initialized with {len(ensemble_models)} active model(s):")
        for n, _, w in ensemble_models:
            print(f"   • {n} (Voting Weight: {w*100:.1f}%)")

    return ensemble_models

def safe_clean_and_convert_audio(audio_input):
    """
    4-Stage Studio Audio Preprocessing:
    1. Converts browser formats (webm, mp4, ogg, wav) to 16kHz mono PCM.
    2. High-pass filter (80 Hz) to eliminate table thumps, AC rumble, desk hum.
    3. Low-pass filter (7600 Hz) to cut out high-frequency digital noise and hiss.
    4. Adaptive FFT Denoising (afftdn) to cancel stationary fan/room noise without muffling speech.
    5. Silence Trimming (VAD): Trims leading dead air so 100% of the window is active Konkani speech.
    """
    if not audio_input:
        return None, 0.0
    out_wav = tempfile.mktemp(suffix=".wav")

    # 4-stage filter chain
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
        # Verify cleaned audio has audible content (>1KB)
        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 1024:
            wf, sr = torchaudio.load(out_wav)
            duration = wf.shape[1] / sr
            return out_wav, duration
        else:
            # Fallback if silence threshold was too aggressive
            fallback_cmd = [
                "ffmpeg", "-y", "-i", str(audio_input),
                "-af", "highpass=f=80,lowpass=f=7600",
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
                out_wav
            ]
            subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            wf, sr = torchaudio.load(out_wav)
            return out_wav, wf.shape[1] / sr
    except Exception as e:
        print("FFmpeg cleaning fallback:", e)
        cmd_basic = [
            "ffmpeg", "-y", "-i", str(audio_input),
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
            out_wav
        ]
        subprocess.run(cmd_basic, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        wf, sr = torchaudio.load(out_wav)
        return out_wav, wf.shape[1] / sr

def predict_voice(audio_file):
    if audio_file is None:
        return (
            "⚠️ **No audio received.** Please record your voice or upload an audio clip.",
            "",
            {},
            None
        )

    models = load_model()
    if not models:
        return (
            f"⚠️ **Checkpoint not found:** No valid model checkpoint files found.",
            "",
            {},
            None
        )

    clean_wav, speech_dur = safe_clean_and_convert_audio(audio_file)
    try:
        waveform, sr = torchaudio.load(clean_wav)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        waveform = waveform.squeeze(0)

        chunk_len = int(MAX_AUDIO_SEC * SAMPLE_RATE)
        stride = int(3.5 * SAMPLE_RATE)  # 3.5s stride with 2.5s overlap

        # Multi-chunk sliding window for long clips (10-20+ seconds)
        chunks = []
        if len(waveform) <= chunk_len:
            pad = chunk_len - len(waveform)
            c = torch.nn.functional.pad(waveform, (0, pad))
            m = torch.cat([torch.ones(len(waveform), dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
            std = c.std()
            if std > 1e-5:
                c = (c - c.mean()) / std
            chunks.append((c, m))
        else:
            start = 0
            while start < len(waveform):
                end = start + chunk_len
                if end <= len(waveform):
                    c = waveform[start:end]
                    m = torch.ones(chunk_len, dtype=torch.long)
                else:
                    pad = chunk_len - (len(waveform) - start)
                    c = torch.nn.functional.pad(waveform[start:], (0, pad))
                    m = torch.cat([torch.ones(len(waveform) - start, dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
                std = c.std()
                if std > 1e-5:
                    c = (c - c.mean()) / std
                chunks.append((c, m))
                if end >= len(waveform):
                    break
                start += stride

        # Ensemble Evaluation across all loaded models
        all_model_probs = []
        model_votes = []

        with torch.no_grad():
            for m_name, net, weight in models:
                chunk_probs = []
                for c, m in chunks:
                    inputs = c.unsqueeze(0).to(device)
                    mask = m.unsqueeze(0).to(device)
                    if hasattr(net, "classifier"):
                        outputs = net(input_values=inputs, attention_mask=mask)
                        d_logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    else:
                        d_logits, _ = net(inputs, mask)
                    p = torch.softmax(d_logits, dim=-1).squeeze(0)
                    chunk_probs.append(p)

                # Average across sliding chunks for this model
                m_avg_p = torch.stack(chunk_probs).mean(dim=0)
                m_pred_idx = int(torch.argmax(m_avg_p))
                m_conf = float(m_avg_p[m_pred_idx]) * 100
                model_votes.append({
                    "name": m_name,
                    "dialect": ID2DIALECT[m_pred_idx],
                    "confidence": m_conf,
                    "weight": weight
                })
                all_model_probs.append(m_avg_p * weight)

        # Final Ensemble Weighted Probability Vector
        probs = torch.stack(all_model_probs).sum(dim=0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        dialect_name = ID2DIALECT[pred_idx]
        confidence = probs[pred_idx] * 100

        region_name, context = REGION_DETAILS.get(
            dialect_name,
            ("Unknown Region", "No regional description available.")
        )

        duration_tip = ""
        if speech_dur < 1.5:
            duration_tip = f"""
            <div style="background: #fff8e1; border-left: 4px solid #ffb300; padding: 10px 14px; border-radius: 6px; font-size: 13px; color: #6d4c41; margin-top: 10px;">
                💡 <b>Short clip detected ({speech_dur:.1f}s):</b> Konkani dialects are cadence-based. For higher confidence, speak 1–2 full sentences!
            </div>
            """

        # Ensemble Badges
        ensemble_badges_html = ""
        if len(models) > 1:
            all_agree = all(v["dialect"] == dialect_name for v in model_votes)
            agree_badge = "🎯 100% Unanimous Consensus" if all_agree else "⚖️ Weighted Model Consensus"
            votes_chips = " &bull; ".join([
                f"<b>{v['name']}</b>: {v['dialect']} ({v['confidence']:.1f}%)"
                for v in model_votes
            ])
            ensemble_badges_html = f"""
            <div style="background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.3); border-radius: 8px; padding: 8px 12px; margin-top: 10px; font-size: 12px;">
                <div style="font-weight: 700; margin-bottom: 3px;">{agree_badge}</div>
                <div style="opacity: 0.95;">{votes_chips}</div>
            </div>
            """
        else:
            single_name = models[0][0]
            ensemble_badges_html = f"""
            <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 6px 10px; margin-top: 10px; font-size: 12px;">
                Active Engine: <b>{single_name}</b>
            </div>
            """

        result_html = f"""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 22px; border-radius: 12px; color: white; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.85;">Detected Konkani Dialect</span>
                <span style="background: rgba(255,255,255,0.25); padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;">✨ Studio Denoised ({speech_dur:.1f}s)</span>
            </div>
            <div style="font-size: 32px; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 8px;">{dialect_name.upper()}</div>
            <div style="display: inline-block; background: rgba(255,255,255,0.2); padding: 5px 14px; border-radius: 20px; font-size: 14px; font-weight: 600;">Confidence: {confidence:.1f}%</div>
            {ensemble_badges_html}
            {duration_tip}
        </div>
        """

        region_html = f"""
        <div style="background: #ffffff; border: 1px solid #e1e8ed; padding: 18px; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-size: 12px; font-weight: 700; color: #5a6e85; text-transform: uppercase; letter-spacing: 1px;">Geographic Origin</div>
            <div style="font-size: 20px; font-weight: 700; color: #1a202c; margin: 4px 0 10px 0;">📍 {region_name}</div>
            <div style="font-size: 14px; line-height: 1.6; color: #4a5568;">{context}</div>
        </div>
        """

        prob_dict = {
            ID2DIALECT[i]: float(probs[i])
            for i in range(len(DIALECT_MAP))
        }

        return result_html, region_html, prob_dict, clean_wav

    except Exception as e:
        return f"⚠️ **Error processing audio:** {str(e)}", "", {}, None

# Gradio Interactive Interface
with gr.Blocks(title="Goan Konkani Dialect Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Goan Konkani Dialect & Region Classifier
        **Speak into your microphone** or upload an audio clip. The audio is automatically **denoised, trimmed of background hum**, and analyzed by the **MMS Neural Classifier** to identify the **Dialect** and **Geographic Origin in Goa**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Voice Input (Microphone or Audio File)"
            )
            submit_btn = gr.Button("🔍 Analyze Voice", variant="primary", size="lg")
            clear_btn = gr.ClearButton([audio_in])

            cleaned_audio_out = gr.Audio(
                label="🔊 Cleaned Speech (Denoised & VAD Trimmed)",
                type="filepath",
                interactive=False
            )

        with gr.Column(scale=1):
            out_dialect = gr.HTML(label="Dialect Result")
            out_region = gr.HTML(label="Geographic Region")
            out_probs = gr.Label(num_top_classes=4, label="Dialect Probabilities")

    submit_btn.click(
        fn=predict_voice,
        inputs=[audio_in],
        outputs=[out_dialect, out_region, out_probs, cleaned_audio_out]
    )

    test_samples_dir = "TEST_AUDIO_SAMPLES"
    if os.path.exists(test_samples_dir):
        sample_files = sorted([
            [os.path.join(test_samples_dir, f)]
            for f in os.listdir(test_samples_dir)
            if f.endswith(".wav")
        ])
        if sample_files:
            gr.Examples(
                examples=sample_files,
                inputs=[audio_in],
                label="🎧 Click Any Sample Below to Test Instantly"
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
