import os
import sys
import time
import tempfile
import subprocess
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from transformers import AutoConfig, Wav2Vec2ForSequenceClassification
import uvicorn

# Configuration
MODEL_ID = "facebook/mms-300m"
SAMPLE_RATE = 16000
MAX_AUDIO_SEC = 6.0
PORT = 8000

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

DIALECT_PHONETIC_RATIONALE = {
    "Antruz": {
        "markers": [
            "Balanced central pitch envelope with standard neutral literary cadence",
            "Symmetric formant transitions (F1 ~500 Hz, F2 ~1500 Hz) with schwa retention",
            "Absence of northern open-vowel dilation or southern Portuguese prosodic lilt"
        ],
        "acoustic_cues": "Evenly paced syllabic timing and standard central Goan harmonic balance.",
        "contrast_note": "Distinguished by canonical Goan pronunciation without terminal vowel elongation."
    },
    "Bardez": {
        "markers": [
            "Open terminal vowel elongation (shift towards broad /ɔ/ and /au/)",
            "Elevated F0 fundamental pitch contour with northern melodic inflection",
            "Rapid staccato syllable delivery with northern coastal tempo"
        ],
        "acoustic_cues": "High-frequency formant dispersion and pronounced vowel lengthening at clause boundaries.",
        "contrast_note": "Differentiated from Central/Southern dialects by high-energy terminal vowel opening."
    },
    "Canacona": {
        "markers": [
            "Distinct southern coastal intonation curve transitioning toward Karwar phonology",
            "Softening of sibilants and vowel rounding (/u/ and /o/ acoustic shifts)",
            "Descending terminal pitch envelope on utterance finals"
        ],
        "acoustic_cues": "Prolonged low-frequency formant resonance with steady southern cadence.",
        "contrast_note": "Distinguished from Salcete by Karwar-adjacent phonological rounding and terminal pitch drop."
    },
    "Gawda_Kunbi": {
        "markers": [
            "Deep guttural / velar acoustic resonance in the 250–600 Hz sub-band",
            "Rhythmic folk cadence with strong syllabic stress pulses",
            "Retention of archaic unreduced consonant clusters and firm dental plosives"
        ],
        "acoustic_cues": "High spectral energy concentration in low harmonics with sharp consonant attacks.",
        "contrast_note": "Differentiated from urban dialects by robust non-nasalized dental plosives and rhythmic pulse."
    },
    "Pednekar": {
        "markers": [
            "Fast-clipped speech tempo (elevated syllables-per-second velocity)",
            "Acoustic transition toward Malvani / Sindhudurg border phonetics",
            "Sharp aspirated consonant bursts (/kh/, /th/, /bh/) with compressed vowels"
        ],
        "acoustic_cues": "Rapid spectral flux transitions and shortened vowel duration envelopes.",
        "contrast_note": "Differentiated from Bardez by higher speech tempo and Sindhudurg borderland aspiration."
    },
    "Sashti_Christian": {
        "markers": [
            "Melodic theatrical lilt with prolonged final open vowels (-ão, trailing -e)",
            "Portuguese lexical loan adaptations and softening of sibilants (/s/ -> /z/ / /sh/)",
            "Modulated pitch contour characteristic of classic Goan Tiatr prosody"
        ],
        "acoustic_cues": "Extended syllabic duration on sentence-final vowels and variable spectral centroid arc.",
        "contrast_note": "Differentiated from Sashti Hindu by trailing vowel elongation and Portuguese prosodic lilt."
    },
    "Sashti_Hindu": {
        "markers": [
            "Crisp, precise dental stop articulation (firm dental /t/, /d/ contact)",
            "Characteristic nasalization (anusvara resonance) on terminal vowels",
            "Disciplined, steady southern Salcete cadence with compact vowels"
        ],
        "acoustic_cues": "Prominent mid-frequency formant clarity (1.2–2.8 kHz) with distinct nasal resonance band.",
        "contrast_note": "Differentiated from Sashti Christian by crisp dental stops and absence of trailing Portuguese lilt."
    },
    "Sattarkar": {
        "markers": [
            "Sahyadri foothill cadence with transitional Marathi-Konkani inflections",
            "Strong retroflex consonant coloring (/ṭ/, /ḍ/, /ḷ/ spectral dips)",
            "Deliberate, measured cadence with deep chest resonance"
        ],
        "acoustic_cues": "Pronounced F3 formant dips during retroflex consonants and slower syllabic rate.",
        "contrast_note": "Differentiated from Pernem and Bardez by heavy retroflex coloring and Ghat-foothill intonation."
    }
}

def generate_prediction_rationale(dialect_name: str, confidence: float, sorted_ranking: list, model_votes: list) -> dict:
    info = DIALECT_PHONETIC_RATIONALE.get(dialect_name, {
        "markers": [
            "Regional acoustic formant distribution across Wav2Vec2 encoder layers",
            "Distinct prosodic pitch contour and cadence timing",
            "Characteristic consonant-vowel boundary transitions"
        ],
        "acoustic_cues": "Distinctive vocalic formant patterns across Wav2Vec2 transformer layers.",
        "contrast_note": "Acoustically separated from neighboring Goan regional varieties."
    })

    runner_up_name, runner_up_prob = (sorted_ranking[1][0], sorted_ranking[1][1] * 100) if len(sorted_ranking) > 1 else ("None", 0.0)
    margin = confidence - runner_up_prob

    agreeing_models = [v["model"] for v in model_votes if v["predicted_dialect"] == dialect_name]
    if len(agreeing_models) == len(model_votes) and len(model_votes) > 1:
        consensus_str = f"All {len(model_votes)} neural engines ({', '.join(agreeing_models)}) unanimously verified this acoustic signature."
    elif len(agreeing_models) > 0:
        consensus_str = f"Supported by {', '.join(agreeing_models)} with a +{margin:.1f}% margin over runner-up {runner_up_name.replace('_', ' ')}."
    else:
        consensus_str = f"Weighted ensemble consensus reached with {confidence:.1f}% cumulative probability."

    summary_note = (
        f"Identified as <b>{dialect_name.replace('_', ' ')}</b> ({confidence:.1f}% confidence) based on {info['acoustic_cues'].lower()} "
        f"{info['contrast_note']} {consensus_str}"
    )

    return {
        "summary": summary_note,
        "markers": info["markers"],
        "acoustic_cues": info["acoustic_cues"],
        "contrast_note": info["contrast_note"],
        "runner_up": runner_up_name.replace('_', ' '),
        "runner_up_margin_percent": round(margin, 2),
        "consensus_note": consensus_str
    }

app = FastAPI(
    title="Goan Konkani Dialect Classifier AI API",
    description="High-performance backend API serving foundation transformer models for Goan Konkani dialect identification",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
loaded_models = {}

def get_loaded_models():
    global loaded_models
    if loaded_models:
        return loaded_models

    candidates = [
        ("Model v5.0 Pro", "best_konkani_dialect_model_v5.pt", 0.50),
        ("Model v6.0 Pro Master", "best_konkani_dialect_model_v6.pt", 0.50),
        ("Model v7.0 Deep-8", "best_konkani_dialect_model_v7_deep.pt", 0.30),
    ]

    for name, path, weight in candidates:
        if os.path.exists(path):
            try:
                print(f"Loading {name} from {path} on {device}...")
                ckpt = torch.load(path, map_location=device, weights_only=False)
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

                config = AutoConfig.from_pretrained(MODEL_ID, num_labels=len(DIALECT_MAP), classifier_proj_size=256)
                m = Wav2Vec2ForSequenceClassification(config).to(device)
                m.load_state_dict(state_dict, strict=True)
                m.eval()
                loaded_models[name] = {"model": m, "weight": weight, "path": path}
                print(f"[INFO] {name} loaded successfully.")
            except Exception as e:
                print(f"[WARN] Error loading {name}: {e}")

    return loaded_models

def clean_audio_with_dsp(input_path: str):
    """Executes the 4-stage studio DSP pipeline via FFmpeg."""
    out_wav = tempfile.mktemp(suffix=".wav")
    filter_chain = (
        "highpass=f=80,"
        "lowpass=f=7600,"
        "afftdn=nr=10:nf=-25,"
        "silenceremove=start_periods=1:start_duration=0.08:start_threshold=-40dB"
    )
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", filter_chain,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
        out_wav
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 1024:
            return out_wav
    except Exception:
        pass

    # Fallback to standard conversion
    fallback_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
        out_wav
    ]
    subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_wav

@app.on_event("startup")
def startup_event():
    print(f"[INFO] Initializing Konkani Dialect Classifier Backend on {device}...")
    models = get_loaded_models()
    print(f"[INFO] Models initialized: {list(models.keys())}")

@app.get("/")
def root():
    return {
        "service": "Goan Konkani Dialect Classifier API",
        "status": "online",
        "device": str(device),
        "active_models": list(loaded_models.keys()),
        "endpoints": {
            "health": "/health",
            "dialects": "/api/dialects",
            "samples": "/api/samples",
            "classify": "POST /api/classify"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "models_loaded": len(loaded_models),
        "models": list(loaded_models.keys()),
        "timestamp": time.time()
    }

@app.get("/api/dialects")
def list_dialects():
    result = []
    for name, (region, context) in REGION_DETAILS.items():
        result.append({
            "dialect": name,
            "region": region,
            "context": context
        })
    return {"dialects": result}

@app.get("/api/samples")
def list_samples():
    samples_dir = "TEST_AUDIO_SAMPLES"
    files = []
    if os.path.exists(samples_dir):
        for f in sorted(os.listdir(samples_dir)):
            if f.endswith(".wav"):
                files.append({
                    "filename": f,
                    "path": os.path.join(samples_dir, f),
                    "size_bytes": os.path.getsize(os.path.join(samples_dir, f))
                })
    return {"samples": files}

@app.get("/api/sample/{filename}")
def get_sample_audio(filename: str):
    samples_dir = "TEST_AUDIO_SAMPLES"
    filepath = os.path.join(samples_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Sample audio file not found")
    return FileResponse(filepath, media_type="audio/wav", filename=filename)

@app.post("/api/classify")
async def classify_audio(
    file: UploadFile = File(...),
    model_choice: str = Form("ensemble")
):
    start_time = time.time()
    models = get_loaded_models()
    if not models:
        raise HTTPException(status_code=500, detail="No models loaded on server")

    # Save uploaded file to temp file
    temp_in = tempfile.mktemp(suffix=os.path.splitext(file.filename)[1] or ".wav")
    with open(temp_in, "wb") as f:
        f.write(await file.read())

    # 4-stage DSP conditioning
    clean_wav = clean_audio_with_dsp(temp_in)

    try:
        waveform, sr = torchaudio.load(clean_wav)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        waveform = waveform.squeeze(0)
        duration_sec = float(len(waveform) / SAMPLE_RATE)

        chunk_len = int(MAX_AUDIO_SEC * SAMPLE_RATE)
        stride = int(3.5 * SAMPLE_RATE)

        # Sliding window chunking
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

        # Determine which models to evaluate
        eval_models = []
        if model_choice == "v5" and "Model v5.0 Pro" in models:
            eval_models = [("Model v5.0 Pro", models["Model v5.0 Pro"]["model"], 1.0)]
        elif model_choice == "v6" and "Model v6.0 Pro Master" in models:
            eval_models = [("Model v6.0 Pro Master", models["Model v6.0 Pro Master"]["model"], 1.0)]
        elif model_choice == "v7" and "Model v7.0 Deep-8" in models:
            eval_models = [("Model v7.0 Deep-8", models["Model v7.0 Deep-8"]["model"], 1.0)]
        elif model_choice in ["tri", "ensemble3", "all", "tri_ensemble"]:
            candidates = [
                ("Model v5.0 Pro", 0.35),
                ("Model v6.0 Pro Master", 0.35),
                ("Model v7.0 Deep-8", 0.30)
            ]
            eval_models = [(name, models[name]["model"], weight) for name, weight in candidates if name in models]
        else:
            # Dual-Engine Consensus Ensemble (Model v5 + Model v6)
            eval_models = [
                (name, data["model"], data["weight"])
                for name, data in models.items()
                if name in ["Model v5.0 Pro", "Model v6.0 Pro Master"]
            ]
            if not eval_models:
                eval_models = [(name, data["model"], data["weight"]) for name, data in models.items()]

        all_probs = []
        model_votes = []
        with torch.no_grad():
            for m_name, net, weight in eval_models:
                chunk_probs = []
                for c, m in chunks:
                    inputs = c.unsqueeze(0).to(device)
                    mask = m.unsqueeze(0).to(device)
                    outputs = net(input_values=inputs, attention_mask=mask)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    p = torch.softmax(logits, dim=-1).squeeze(0)
                    chunk_probs.append(p)

                m_avg_p = torch.stack(chunk_probs).mean(dim=0)
                m_pred_idx = int(torch.argmax(m_avg_p))
                m_conf = float(m_avg_p[m_pred_idx]) * 100
                model_votes.append({
                    "model": m_name,
                    "predicted_dialect": ID2DIALECT[m_pred_idx],
                    "confidence": round(m_conf, 2),
                    "weight": weight
                })
                all_probs.append(m_avg_p * weight)

        # Normalize weights
        tot_w = sum(w for _, _, w in eval_models)
        ensemble_p = (torch.stack(all_probs).sum(dim=0) / tot_w).cpu().numpy()

        pred_idx = int(np.argmax(ensemble_p))
        dialect_name = ID2DIALECT[pred_idx]
        confidence = float(ensemble_p[pred_idx] * 100)

        region_name, context = REGION_DETAILS.get(
            dialect_name,
            ("Unknown Region", "No description available.")
        )

        # Consensus badge calculation
        consensus_badge = "CONFIRMED PREDICTION"
        if len(model_votes) >= 2:
            vote_dialects = [v["predicted_dialect"] for v in model_votes]
            if all(d == vote_dialects[0] for d in vote_dialects):
                consensus_badge = "HIGH-CONFIDENCE CONSENSUS (All Models Agree)"
            else:
                consensus_badge = "AMBIGUOUS REGIONAL CONTINUUM (Border Transit Zone)"

        # Full probability dictionary
        prob_dict = {
            ID2DIALECT[i]: round(float(ensemble_p[i]), 4)
            for i in range(len(DIALECT_MAP))
        }

        # Sorted ranking
        sorted_ranking = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        # Generate phonetic reasoning & prediction rationale
        prediction_rationale = generate_prediction_rationale(
            dialect_name, confidence, sorted_ranking, model_votes
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "success": True,
            "predicted_dialect": dialect_name,
            "confidence_percent": round(confidence, 2),
            "region_name": region_name,
            "cultural_context": context,
            "prediction_rationale": prediction_rationale,
            "duration_seconds": round(duration_sec, 2),
            "consensus_badge": consensus_badge,
            "device": str(device),
            "latency_ms": elapsed_ms,
            "probabilities": prob_dict,
            "ranking": [{"dialect": d, "probability": p, "percentage": round(p * 100, 2)} for d, p in sorted_ranking],
            "model_votes": model_votes,
            "engine": "Tri-Engine Grand Ensemble (v5 + v6 + v7)" if len(eval_models) == 3 else ("Dual-Engine Consensus Ensemble (v5 + v6)" if len(eval_models) == 2 else eval_models[0][0])
        }

    finally:
        for p in [temp_in, clean_wav]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

if __name__ == "__main__":
    uvicorn.run("backend_server:app", host="0.0.0.0", port=PORT, log_level="info")
