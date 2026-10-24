# Goan Konkani Dialect Classifier: Java Frontend & AI Microservice Architecture

This subsystem fulfills the academic requirement for a **complete Frontend in Java** integrated seamlessly with our **Acoustic-First PyTorch Deep Neural Foundation Model (Meta MMS-300M)**.

---

## 🏛️ System Architecture Overview

The system is designed following the enterprise-standard **Microservice Decoupled Pattern**:

```
+-------------------------------------------------------------+
|                JAVA 17 DESKTOP GUI FRONTEND                 |
|             (KonkaniDialectClassifierApp.java)              |
|                                                             |
|  [📁 File Browser]   [🎧 Pre-Loaded Samples]   [🔴 Live Mic] |
|                                                             |
|  [▶️ Audio Player]   [⚙️ Model Engine]       [🚀 Classify]  |
|                                                             |
|  [🎯 Dialect Card]   [📍 Geographic Origin]   [📊 Bar Chart] |
+-------------------------------------------------------------+
                              |
               HTTP REST / Multipart Audio Upload
               (Native Java 17 HttpClient)
                              |
                              v
+-------------------------------------------------------------+
|             PYTHON FASTAPI AI MODEL BACKEND                |
|                   (backend_server.py)                       |
|                                                             |
|  • 4-Stage Studio DSP Preprocessing (FFmpeg HPF, LPF, VAD)  |
|  • Sliding-Window Multi-Chunk Temporal Inference Engine     |
|  • Apple Silicon M4 Metal Performance Shaders (MPS)         |
|  • Model v5.0 Pro + Model v6.0 Pro Master + v7.0 Deep-8     |
|  • Soft-Voting Dual-Engine Consensus Ensemble Logic         |
+-------------------------------------------------------------+
```

---

## 🚀 Quick Start: Launching the Java Frontend

### One-Click Launch
In your terminal, simply run:
```bash
./run_java_app.sh
```
This script automatically:
1. Verifies that the Python AI model server is active on `http://localhost:8000` (starts it if needed).
2. Verifies that the models are loaded into Apple Silicon GPU memory (`mps`).
3. Compiles the Java source files using `javac`.
4. Opens the **Java Desktop GUI Window** directly on your screen!

---

## 💻 Manual Commands

### 1. Recompile Java Source Code & Rebuild JAR
```bash
./compile_java.sh
```
Or directly:
```bash
javac KonkaniDialectClassifierApp.java KonkaniDialectCLI.java
jar cfe KonkaniDialectClassifier.jar KonkaniDialectClassifierApp KonkaniDialectClassifierApp*.class
```

### 2. Run the Java Desktop GUI Application
```bash
java -jar KonkaniDialectClassifier.jar
```
*(Or `java KonkaniDialectClassifierApp`)*

### 3. Run the Java Command-Line Client (CLI)
You can classify any audio clip directly from the terminal using the Java CLI:
```bash
java KonkaniDialectCLI TEST_AUDIO_SAMPLES/01_Bardez_NorthGoa.wav
```
Or specify a particular model:
```bash
java KonkaniDialectCLI TEST_AUDIO_SAMPLES/02_Sashti_Christian_SouthGoa.wav v6
```

---

## 🎨 Java Desktop GUI Features

1. **Pre-Loaded Test Audio Dropdown:**
   * Contains real Goan broadcast speech samples from `TEST_AUDIO_SAMPLES/` (*Bardez, Sashti Christian, Sashti Hindu, Antruz, Canacona, Sattarkar, Pernem, Gawda Kunbi, Prudent News, Goan Tiatr*).
   * Allows instant one-click testing without needing external files!

2. **File Browser (WAV, MP3, OGG, FLAC):**
   * Select any recorded speech clip from your computer.

3. **Live Microphone Recording (`javax.sound.sampled`):**
   * Native Java audio recording at 16 kHz Mono PCM.
   * Click **"🔴 Start Recording"**, speak into your microphone, click **"⏹️ Stop Recording"**, and classify your own voice!

4. **Built-in Audio Player:**
   * Listen to the selected audio sample or your recorded voice directly inside Java before running inference.

5. **Model Selection:**
   * **🌟 Dual-Engine Consensus Ensemble (Model v5.0 + Model v6.0)** *(Recommended: 84.1% Top-2 accuracy)*
   * **🔷 Model v6.0 Pro Master** *(Trained on 7,628 Master dataset clips)*
   * **🔶 Model v5.0 Pro** *(High-precision phonetic boundary specialized)*
   * **🟣 Model v7.0 Deep-8** *(Experimental deep unfreezing)*

6. **Interactive Graphical Visualizations:**
   * **Result Card:** Large bold dialect name with confidence badge.
   * **Consensus Badge:** Displays `HIGH-CONFIDENCE CONSENSUS` when models agree, or flags border transit zones (`AMBIGUOUS REGIONAL CONTINUUM`).
   * **Geographic Origin Card:** Identifies the taluka, village cluster, and cultural sociolinguistic context in Goa.
   * **Custom Horizontal Bar Chart:** Custom-painted Java graphic chart rendering the exact probability percentage for all 8 dialects.
   * **System Telemetry:** Real-time latency measurement in milliseconds and hardware compute device (`Apple Silicon M4 MPS`).

---

## 🌐 FastAPI Backend Endpoints

The Python server runs at `http://localhost:8000`:
* Interactive Swagger API Documentation: **`http://localhost:8000/docs`**
* Health Check: **`GET /health`**
* Dialect Metadata: **`GET /api/dialects`**
* Available Test Samples: **`GET /api/samples`**
* Dialect Classification: **`POST /api/classify`** (Accepts `multipart/form-data` with audio file and model choice parameter)
