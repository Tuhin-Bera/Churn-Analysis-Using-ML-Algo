# Deployment Guide — Customer Lifetime Value & Churn Intelligence System

This document covers every supported deployment method: local development, Docker (single container + full PostgreSQL stack), Streamlit Cloud, Heroku, Railway, and Render.

---

## Prerequisites

| Requirement | Minimum Version |
|---|---|
| Python | 3.11 |
| pip | 23+ |
| Docker (optional) | 24+ |
| PostgreSQL (optional) | 14+ |

---

## 1. Local Development (Quickest)

```bash
# Clone / open the project
cd "part 3"

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env            # then edit .env with your settings

# Train all models (creates outputs/ directory)
python train_pipeline.py

# Launch the dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 2. Docker — Single Container (SQLite)

No external database needed. SQLite is used automatically.

```bash
# Build the image
docker build -t churn-intelligence .

# Run (training happens automatically on first start)
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/outputs:/app/outputs \
  churn-intelligence
```

Open **http://localhost:8501**.

---

## 3. Docker Compose — Full Stack (App + PostgreSQL)

```bash
# Copy and configure environment
cp .env.example .env
# Edit PG_PASSWORD in .env

# Start both services
docker compose up --build

# Stop
docker compose down

# Stop and remove DB volume (WARNING: deletes all data)
docker compose down -v
```

PostgreSQL is exposed on `localhost:5432`.
Streamlit runs on **http://localhost:8501**.

---

## 4. Streamlit Cloud (Free, Recommended for Portfolio)

1. Push this repository to **GitHub** (public or private).
2. Go to **https://streamlit.io/cloud** → **New app**.
3. Select your repo, branch `master`, and set **Main file path** to `app.py`.
4. Click **Advanced settings → Secrets** and paste the contents of `.streamlit/secrets.toml`.
5. Click **Deploy** — Streamlit Cloud installs dependencies from `requirements.txt` automatically.

> **Note:** Streamlit Cloud has no persistent disk. The training pipeline must produce outputs that are either committed to git or regenerated on each cold start. We recommend committing `outputs/master_predictions.csv` and `outputs/lgbm_churn_model.pkl` to the repo (or using a cloud PostgreSQL database via `DATABASE_URL`).

---

## 5. Heroku

```bash
# Install Heroku CLI and log in
heroku login

# Create app
heroku create churn-intelligence-app

# Set secrets (never commit .env to git)
heroku config:set PG_HOST=<host> PG_PORT=5432 \
  PG_DATABASE=telecom_churn PG_USER=<user> PG_PASSWORD=<pass>

# Add PostgreSQL add-on
heroku addons:create heroku-postgresql:mini

# Deploy
git push heroku master

# Check logs
heroku logs --tail
```

The `Procfile` and `setup.sh` handle training and startup automatically.

---

## 6. Railway

1. Go to **https://railway.app** → **New Project → Deploy from GitHub repo**.
2. Select this repository.
3. Railway auto-detects the `Procfile`. Set environment variables in the **Variables** tab using values from `.env.example`.
4. Add a **PostgreSQL** plugin from the Railway dashboard.
5. Copy the `DATABASE_URL` from the plugin into the Variables tab.
6. Click **Deploy**.

---

## 7. Render

1. Go to **https://render.com** → **New Web Service**.
2. Connect your GitHub repository.
3. Set **Build Command:** `pip install -r requirements.txt && python train_pipeline.py`
4. Set **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
5. Set environment variables in the **Environment** tab (copy from `.env.example`).
6. Click **Create Web Service**.

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `PG_HOST` | `localhost` | PostgreSQL host (app falls back to SQLite if unreachable) |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_DATABASE` | `telecom_churn` | Database name |
| `PG_USER` | `postgres` | Database user |
| `PG_PASSWORD` | *(none)* | Database password — **keep secret** |
| `DATABASE_URL` | *(derived)* | Full connection string; overrides individual PG_* vars |
| `APP_ENV` | `development` | `development` or `production` |
| `DATA_PATH` | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Path to raw CSV |
| `OUTPUTS_DIR` | `outputs` | Directory for model artifacts and predictions |

---

## File Reference

| File | Purpose |
|---|---|
| `.env` | Local secrets (gitignored) |
| `.env.example` | Safe template committed to git |
| `.gitignore` | Prevents secrets and binaries from being committed |
| `.streamlit/config.toml` | Streamlit server, theme, and logger settings |
| `.streamlit/secrets.toml` | Streamlit Cloud secrets (gitignored) |
| `Procfile` | Heroku / Railway process declaration |
| `setup.sh` | Pre-start script (trains models if absent) |
| `runtime.txt` | Python version pin for Heroku |
| `Dockerfile` | Single-container Docker build |
| `.dockerignore` | Excludes files from Docker build context |
| `docker-compose.yml` | Full local stack with PostgreSQL |
| `pyproject.toml` | PEP 517 build system and tool config |
| `requirements.txt` | Production Python dependencies |
| `requirements-dev.txt` | Development and testing extras |
