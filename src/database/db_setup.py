"""
Database Setup and Management Module
=====================================

This module handles all PostgreSQL database operations:
- Creating tables for customer data, predictions, and segments
- Loading CSV data into the database
- Fetching data back for modeling and the Streamlit app

NOTE: PostgreSQL connection string is set in config.py
      If PostgreSQL is unavailable, falls back to SQLite for demo purposes.
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, Column, String, Float, Integer, Boolean
from sqlalchemy.exc import OperationalError
import warnings
warnings.filterwarnings("ignore")

# --- connection helper --------------------------------------------------------

def get_engine(use_sqlite_fallback: bool = True):
    """
    Returns a SQLAlchemy engine.
    Tries PostgreSQL first; if that fails and fallback=True, uses SQLite.
    
    We learned from experience: always have a fallback for demos!
    """
    pg_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/telecom_churn"
    )
    try:
        engine = create_engine(pg_url, pool_pre_ping=True)
        # quick connection test
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[DB] Connected to PostgreSQL successfully.")
        return engine
    except Exception as e:
        if use_sqlite_fallback:
            print(f"[DB] PostgreSQL unavailable ({type(e).__name__}). Falling back to SQLite.")
            sqlite_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "outputs", "telecom_churn.db"
            )
            os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
            engine = create_engine(f"sqlite:///{sqlite_path}")
            print(f"[DB] Using SQLite at: {sqlite_path}")
            return engine
        else:
            raise


# --- schema creation ----------------------------------------------------------

CREATE_TABLES_SQL = """
-- Raw customer data as imported from the CSV
CREATE TABLE IF NOT EXISTS customers (
    customer_id         TEXT PRIMARY KEY,
    gender              TEXT,
    senior_citizen      INTEGER,
    partner             TEXT,
    dependents          TEXT,
    tenure              INTEGER,
    phone_service       TEXT,
    multiple_lines      TEXT,
    internet_service    TEXT,
    online_security     TEXT,
    online_backup       TEXT,
    device_protection   TEXT,
    tech_support        TEXT,
    streaming_tv        TEXT,
    streaming_movies    TEXT,
    contract            TEXT,
    paperless_billing   TEXT,
    payment_method      TEXT,
    monthly_charges     REAL,
    total_charges       REAL,
    churn               TEXT
);

-- Model predictions: churn probability, CLV, survival estimate
CREATE TABLE IF NOT EXISTS predictions (
    customer_id         TEXT PRIMARY KEY,
    churn_probability   REAL,
    churn_prediction    INTEGER,
    clv_estimate        REAL,
    survival_months     REAL,
    uplift_score        REAL,
    customer_segment    TEXT
);

-- Segment summary for reporting
CREATE TABLE IF NOT EXISTS segments (
    segment_id          INTEGER PRIMARY KEY,
    segment_name        TEXT,
    avg_monthly_charges REAL,
    avg_tenure          REAL,
    avg_clv             REAL,
    churn_rate          REAL,
    customer_count      INTEGER
);

-- ROI campaign simulation log
CREATE TABLE IF NOT EXISTS campaign_log (
    campaign_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at              TEXT,
    discount_amount     REAL,
    num_customers       INTEGER,
    expected_retained   INTEGER,
    total_cost          REAL,
    expected_benefit    REAL,
    roi_pct             REAL
);
"""

# SQLite doesn't support AUTOINCREMENT in the same way; patch it
CREATE_TABLES_SQL_SQLITE = CREATE_TABLES_SQL.replace(
    "INTEGER PRIMARY KEY AUTOINCREMENT",
    "INTEGER PRIMARY KEY"
)


def create_tables(engine):
    """Create all tables if they don't already exist."""
    sql = CREATE_TABLES_SQL_SQLITE  # works for both SQLite & modern PostgreSQL
    with engine.begin() as conn:
        for statement in sql.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
    print("[DB] Tables created / verified.")


# --- data loading -------------------------------------------------------------

def load_csv_to_db(csv_path: str, engine, if_exists: str = "replace") -> int:
    """
    Reads the raw Telco CSV, cleans column names, and writes to 'customers' table.
    Returns number of rows loaded.
    """
    df = pd.read_csv(csv_path)

    # --- light cleaning ---
    # TotalCharges has some spaces where tenure=0, coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # For rows where TotalCharges is NaN, tenure is 0 => fill with MonthlyCharges
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"]

    # Rename columns to snake_case to match our schema
    rename_map = {
        "customerID": "customer_id",
        "gender": "gender",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "tenure": "tenure",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "Contract": "contract",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Churn": "churn",
    }
    df = df.rename(columns=rename_map)

    # Write to database
    df.to_sql("customers", engine, if_exists=if_exists, index=False)
    print(f"[DB] Loaded {len(df)} customer records into 'customers' table.")
    return len(df)


def fetch_customers(engine) -> pd.DataFrame:
    """Pull all customers from the database."""
    return pd.read_sql("SELECT * FROM customers", engine)


def save_predictions(df_pred: pd.DataFrame, engine):
    """Upsert model predictions back to database."""
    df_pred.to_sql("predictions", engine, if_exists="replace", index=False)
    print(f"[DB] Saved {len(df_pred)} prediction rows.")


def save_segments(df_seg: pd.DataFrame, engine):
    """Save segment summary statistics."""
    df_seg.to_sql("segments", engine, if_exists="replace", index=False)
    print(f"[DB] Saved {len(df_seg)} segment rows.")


def log_campaign(engine, discount: float, n_customers: int,
                 expected_retained: int, benefit: float):
    """Append a campaign simulation run to the log."""
    from datetime import datetime
    cost = discount * n_customers
    roi = (benefit - cost) / cost * 100 if cost > 0 else 0.0
    row = {
        "run_at": datetime.now().isoformat(),
        "discount_amount": discount,
        "num_customers": n_customers,
        "expected_retained": expected_retained,
        "total_cost": cost,
        "expected_benefit": benefit,
        "roi_pct": roi,
    }
    pd.DataFrame([row]).to_sql("campaign_log", engine, if_exists="append", index=False)
    print(f"[DB] Campaign logged. ROI={roi:.1f}%")


# --- quick test ---------------------------------------------------------------

if __name__ == "__main__":
    engine = get_engine()
    create_tables(engine)
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    if os.path.exists(csv_path):
        n = load_csv_to_db(csv_path, engine)
        df = fetch_customers(engine)
        print(f"Fetched {len(df)} rows back. Columns: {df.columns.tolist()}")
