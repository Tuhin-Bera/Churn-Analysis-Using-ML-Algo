#!/bin/bash
# ==============================================================================
# setup.sh — Pre-start script for Render (and Heroku / Railway)
# ==============================================================================
# Render runs this script before starting the web process defined in Procfile.
# It guarantees that:
#   1. Required output directories exist.
#   2. The training pipeline runs on cold-start (first deploy) when model
#      artifacts are absent. Subsequent deploys skip training entirely,
#      keeping restarts fast.
#
# On Render the working directory is the repo root, so all relative paths
# resolve correctly against the committed project structure.
# ==============================================================================

set -e  # abort immediately on any non-zero exit code

echo "========================================================"
echo "  Churn Intelligence System — Render Pre-Start Setup"
echo "========================================================"

# Create runtime directories that are gitignored
mkdir -p outputs
mkdir -p data

# Run the full training pipeline only when the primary model artifact is absent.
# This handles the very first cold-start on a fresh Render deployment.
if [ ! -f "outputs/lgbm_churn_model.pkl" ]; then
    echo "--> Model artifacts not found. Running training pipeline (~15s)..."
    python train_pipeline.py
    echo "--> Training complete."
else
    echo "--> Model artifacts already present. Skipping training."
fi

echo "========================================================"
echo "  Setup complete. Launching Streamlit..."
echo "========================================================"
