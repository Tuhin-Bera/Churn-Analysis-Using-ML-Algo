"""
Database Setup and Management Module
=====================================

This module handles all database persistence operations using SQLAlchemy:
- Establishing a reliable connection to PostgreSQL (with an automatic SQLite fallback)
- Creating relational schemas for customer records, model predictions, segments, and campaign simulations
- Ingesting raw customer records from CSV into structured database tables
- Exporting and reading records for both batch model training and interactive Streamlit querying

Why this matters:
In production data science systems, models should not simply read raw CSV files. Storing customer
profiles, churn probabilities, and campaign simulations in an ACID-compliant database enables
auditability, scheduled re-training, and integration with CRM platforms.
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, Column, String, Float, Integer, Boolean
from sqlalchemy.exc import OperationalError
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Database Connection Helper
# -----------------------------------------------------------------------------

def get_engine(use_sqlite_fallback: bool = True):
    """
    Returns an active SQLAlchemy engine.
    
    Design Decision:
    We attempt connecting to PostgreSQL first using standard environment variables
    or a default local connection URI. If PostgreSQL is offline or credentials fail,
    we smoothly fall back to a local SQLite database in the 'outputs' directory.
    
    This ensures the pipeline and Streamlit app work out-of-the-box on any reviewer machine
    without requiring manual local PostgreSQL configuration.
    """
    pg_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/telecom_churn"
    )
    try:
        # pool_pre_ping checks if the existing database connection is still alive before querying
        engine = create_engine(pg_url, pool_pre_ping=True)
        # Verify connection with a lightweight probe query
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[DB] Connected to PostgreSQL successfully.")
        return engine
    except Exception as e:
        if use_sqlite_fallback:
            print(f"[DB] PostgreSQL unavailable ({type(e).__name__}). Falling back to SQLite.")
            # Resolve directory path relative to project structure
            sqlite_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "outputs", "telecom_churn.db"
            )
            os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
            engine = create_engine(f"sqlite:///{sqlite_path}")
            print(f"[DB] Using SQLite fallback database at: {sqlite_path}")
            return engine
        else:
            raise


# -----------------------------------------------------------------------------
# Relational Schema Definition
# -----------------------------------------------------------------------------

# We define 4 relational tables:
# 1. customers: stores historical telecom features
# 2. predictions: stores scored churn probabilities, predicted survival months, and CLV
# 3. segments: stores cluster level aggregates and business playbooks
# 4. campaign_log: tracks historical simulation runs from the ROI simulator
CREATE_TABLES_SQL = """
-- Table 1: Raw customer baseline features
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

-- Table 2: Model predictions & individual scoring
CREATE TABLE IF NOT EXISTS predictions (
    customer_id         TEXT PRIMARY KEY,
    churn_probability   REAL,
    churn_prediction    INTEGER,
    clv_estimate        REAL,
    survival_months     REAL,
    uplift_score        REAL,
    customer_segment    TEXT
);

-- Table 3: Segment summaries for high-level dashboard metrics
CREATE TABLE IF NOT EXISTS segments (
    segment_id          INTEGER PRIMARY KEY,
    segment_name        TEXT,
    avg_monthly_charges REAL,
    avg_tenure          REAL,
    avg_clv             REAL,
    churn_rate          REAL,
    customer_count      INTEGER
);

-- Table 4: Marketing simulation history and audit trail
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

# SQLite treats AUTOINCREMENT differently than PostgreSQL, so we adapt syntax cleanly
CREATE_TABLES_SQL_SQLITE = CREATE_TABLES_SQL.replace(
    "INTEGER PRIMARY KEY AUTOINCREMENT",
    "INTEGER PRIMARY KEY"
)


def create_tables(engine):
    """
    Executes schema creation statements against the database engine.
    Splits multi-statement blocks and executes sequentially in a transactional block.
    """
    sql = CREATE_TABLES_SQL_SQLITE
    with engine.begin() as conn:
        for statement in sql.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
    print("[DB] Tables created / verified successfully.")


# -----------------------------------------------------------------------------
# Data Ingestion & Querying Utilities
# -----------------------------------------------------------------------------

def load_csv_to_db(csv_path: str, engine, if_exists: str = "replace") -> int:
    """
    Reads the raw Telco CSV, handles known anomalies (like whitespace TotalCharges),
    standardizes column headers to snake_case, and loads records into the 'customers' table.
    
    Returns:
        int: Number of rows successfully inserted.
    """
    df = pd.read_csv(csv_path)

    # Data Quality Fix:
    # In the raw dataset, TotalCharges is stored as a string and contains blank spaces ' '
    # for brand new customers where tenure == 0.
    # Coercing to numeric turns those blanks into NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # For new joiners with tenure == 0, their total charges equal their first month's charges
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"]

    # Map raw CSV PascalCase headers into clean database snake_case
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

    # Persist directly via SQLAlchemy
    df.to_sql("customers", engine, if_exists=if_exists, index=False)
    print(f"[DB] Loaded {len(df)} customer records into 'customers' table.")
    return len(df)


def fetch_customers(engine) -> pd.DataFrame:
    """Retrieves all baseline customer records from the database."""
    return pd.read_sql("SELECT * FROM customers", engine)


def save_predictions(df_pred: pd.DataFrame, engine):
    """Saves scored model predictions into the 'predictions' table."""
    df_pred.to_sql("predictions", engine, if_exists="replace", index=False)
    print(f"[DB] Saved {len(df_pred)} prediction rows.")


def save_segments(df_seg: pd.DataFrame, engine):
    """Saves high-level segment statistics into the 'segments' table."""
    df_seg.to_sql("segments", engine, if_exists="replace", index=False)
    print(f"[DB] Saved {len(df_seg)} segment rows.")


def log_campaign(engine, discount: float, n_customers: int,
                 expected_retained: int, benefit: float):
    """
    Logs an interactive campaign simulation run from the Streamlit ROI Simulator.
    Calculates total cost and net percentage ROI to provide an audit trail of marketing tests.
    """
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
    print(f"[DB] Campaign logged: discount=Rs.{discount}, cost=Rs.{cost:,.0f}, ROI={roi:.1f}%")


# -----------------------------------------------------------------------------
# Quick Standalone Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Test connection and table creation
    engine = get_engine()
    create_tables(engine)
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    if os.path.exists(csv_path):
        n = load_csv_to_db(csv_path, engine)
        df = fetch_customers(engine)
        print(f"[DB Test] Read back {len(df)} rows. Sample columns: {df.columns.tolist()[:6]}")
