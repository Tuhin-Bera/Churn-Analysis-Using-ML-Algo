# ==============================================================================
# Dockerfile — Customer Lifetime Value & Churn Intelligence System
# ==============================================================================
# Builds a self-contained Docker image that:
#   1. Installs all Python dependencies
#   2. Copies the full project into the container
#   3. Runs the training pipeline on first start (if outputs/ is empty)
#   4. Launches the Streamlit web application
#
# Build:
#   docker build -t churn-intelligence .
#
# Run (standalone, SQLite):
#   docker run -p 8501:8501 churn-intelligence
#
# Run (with .env file):
#   docker run -p 8501:8501 --env-file .env churn-intelligence
# ==============================================================================

# ---- Base image -------------------------------------------------------
FROM python:3.11-slim

# Set labels for documentation
LABEL description="Customer Lifetime Value and Churn Intelligence System"
LABEL maintainer="Nexvia Technologies — Data Science Track"

# ---- OS dependencies --------------------------------------------------
# build-essential is needed to compile some native extensions (e.g. lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory ------------------------------------------------
WORKDIR /app

# ---- Python dependencies (cached layer, only re-run if requirements change) ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- Copy application source -----------------------------------------
COPY . .

# ---- Create necessary directories (outputs/ may not exist on fresh clone) ---
RUN mkdir -p outputs data assets

# ---- Expose Streamlit default port ------------------------------------
EXPOSE 8501

# ---- Healthcheck (platforms like Render use this) ---------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ---- Entrypoint -------------------------------------------------------
# setup.sh runs the training pipeline if model artifacts are absent,
# then hands off to Streamlit.
ENTRYPOINT ["bash", "setup.sh"]
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
