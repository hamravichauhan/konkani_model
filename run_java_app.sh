#!/usr/bin/env bash
# ==============================================================================
# Goan Konkani Dialect Classifier - One-Click Launcher for Java Frontend
# Launches Python FastAPI Model Backend and Opens Java Swing GUI Application
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=============================================================================="
echo "  GOAN KONKANI DIALECT IDENTIFICATION USING SPEECH AUDIO - LAUNCHER"
echo "=============================================================================="

# 1. Check Python Virtual Environment
PYTHON_BIN="./venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    echo "[WARN] Local virtual environment not found at ./venv/bin/python, using system python3..."
    PYTHON_BIN="python3"
fi

# 2. Check if Backend Server is already running on port 8000
BACKEND_RUNNING=false
if curl -s http://localhost:8000/health | grep -q "status"; then
    echo "[INFO] Python Model Backend Server is already running on port 8000."
    BACKEND_RUNNING=true
fi

if [ "$BACKEND_RUNNING" = false ]; then
    echo "[INFO] Starting Python AI Model Backend Server (FastAPI + MMS-300M)..."
    nohup $PYTHON_BIN backend_server.py > backend_server.log 2>&1 &
    SERVER_PID=$!
    echo "   Backend Process PID: $SERVER_PID (Logging to backend_server.log)"

    echo "[INFO] Waiting for models to load on Apple Silicon MPS / GPU..."
    RETRIES=0
    MAX_RETRIES=40
    while [ $RETRIES -lt $MAX_RETRIES ]; do
        if curl -s http://localhost:8000/health | grep -q "status"; then
            echo "[INFO] AI Model Backend is online and models are loaded!"
            break
        fi
        sleep 1
        RETRIES=$((RETRIES + 1))
        echo -n "."
    done
    echo ""

    if [ $RETRIES -ge $MAX_RETRIES ]; then
        echo "[ERROR] Backend server failed to start within 40 seconds. Check backend_server.log:"
        tail -n 20 backend_server.log
        exit 1
    fi
fi

# 3. Ensure Java binaries are compiled
if [ ! -f "KonkaniDialectClassifier.jar" ]; then
    echo "[BUILD] Compiling Java Application..."
    ./compile_java.sh
fi

# 4. Launch Java Desktop GUI
echo "[LAUNCH] Launching Java 17 Swing Desktop GUI Application..."
java -Dawt.useSystemAAFontSettings=on -Dswing.aatext=true -Dsun.java2d.renderer=sun.java2d.marlin.MarlinRenderingEngine -jar KonkaniDialectClassifier.jar
