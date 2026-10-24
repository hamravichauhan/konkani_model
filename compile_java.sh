#!/usr/bin/env bash
# ==============================================================================
# Goan Konkani Dialect Classifier - Java Compilation & Packaging Script
# ==============================================================================
set -e

echo "[BUILD] Compiling Java Frontend and CLI Client using $(javac -version 2>&1)..."
javac KonkaniDialectClassifierApp.java KonkaniDialectCLI.java

echo "[PACKAGE] Packaging executable standalone JAR: KonkaniDialectClassifier.jar..."
jar cfe KonkaniDialectClassifier.jar KonkaniDialectClassifierApp KonkaniDialectClassifierApp*.class

echo "[SUCCESS] Compilation successful!"
echo "   • GUI Application JAR: KonkaniDialectClassifier.jar"
echo "   • GUI Class:           KonkaniDialectClassifierApp.class"
echo "   • CLI Class:           KonkaniDialectCLI.class"
