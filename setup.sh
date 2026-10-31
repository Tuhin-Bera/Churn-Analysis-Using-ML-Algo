#!/bin/bash
# ==============================================================================
# setup.sh — Pre-start setup script for Heroku / Render / Railway
# ==============================================================================
# This script runs once before the app server starts.
# It ensures the outputs directory exists and runs the training pipeline
# if model artifacts are missing (cold-start on a fresh deployment).

set -e  # exit on any error

echo "==> Creating required directories..."
mkdir -p outputs
mkdir -p data

# Run the training pipeline only if the model artifact is missing.
# This handles a fresh deployment where outputs/ is empty.
if [ ! -f "outputs/lgbm_churn_model.pkl" ]; then
    echo "==> Model artifacts not found. Running training pipeline..."
    python train_pipeline.py
    echo "==> Training pipeline complete."
else
    echo "==> Model artifacts already present. Skipping training."
fi

echo "==> Setup complete. Starting Streamlit..."
