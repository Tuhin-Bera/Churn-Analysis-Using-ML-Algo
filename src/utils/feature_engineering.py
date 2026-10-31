"""
Feature Engineering & Preprocessing Module
============================================

This is where the data transformation magic happens.
We spent a lot of time thinking about which features actually make sense
from a business perspective, not just from a pure ML standpoint.

Key ideas:
- Tenure groups: short/medium/long tenure tells a lot about loyalty
- Revenue per month ratio: how much does tenure explain total charges?
- Service count: customers with more services may be stickier
- Contract risk score: month-to-month = highest risk
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# --- raw data loading ---------------------------------------------------------

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw Telco CSV with basic type fixes."""
    df = pd.read_csv(filepath)

    # TotalCharges is a string column with some spaces — annoying but common in real data
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # For new customers (tenure=0), TotalCharges is missing -> fill with MonthlyCharges
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # Churn to binary integer — easier to work with
    df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)

    return df


# --- feature engineering ------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates additional features on top of the raw columns.
    
    We derived these after doing EDA — each one has a business justification.
    """
    df = df.copy()

    # --- Tenure groups ---
    # Customers who've been around for less than a year are the riskiest
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72, np.inf],
        labels=["0-1 yr", "1-2 yr", "2-4 yr", "4-6 yr", "6+ yr"],
        right=True,
        include_lowest=True
    )

    # --- Revenue consistency: monthly vs total ---
    # If total_charges / (tenure+1) ≈ monthly_charges, customer is consistent
    df["avg_monthly_charge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["charge_ratio"] = df["MonthlyCharges"] / (df["avg_monthly_charge"] + 1e-6)

    # --- Service count: how many add-ons does the customer have? ---
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    # Count services that are "Yes" (not "No" or "No internet service")
    df["service_count"] = df[service_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    # --- Contract risk: numeric encoding of contract duration ---
    contract_risk = {
        "Month-to-month": 3,  # highest risk
        "One year": 2,
        "Two year": 1          # lowest risk (committed)
    }
    df["contract_risk"] = df["Contract"].map(contract_risk)

    # --- Senior citizen flag is already 0/1 but let's keep it explicit ---
    df["is_senior"] = df["SeniorCitizen"]

    # --- Payment method risk ---
    # Electronic check customers tend to churn more (observed in EDA)
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    # --- Has internet service ---
    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    # --- Paperless billing flag ---
    df["paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)

    # --- Partner / Dependents -> family flag ---
    df["has_family"] = (
        (df["Partner"] == "Yes") | (df["Dependents"] == "Yes")
    ).astype(int)

    # --- High spender flag ---
    df["is_high_spender"] = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)

    return df


# --- encoding for ML ----------------------------------------------------------

# These are the features we will feed to LightGBM
FEATURE_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_charge",
    "charge_ratio",
    "service_count",
    "contract_risk",
    "is_senior",
    "is_electronic_check",
    "has_internet",
    "paperless",
    "has_family",
    "is_high_spender",
    # One-hot encoded columns added after preprocessing
    "gender_Male",
    "InternetService_Fiber optic",
    "InternetService_No",
]

CATEGORICAL_COLS = ["gender", "InternetService", "PaymentMethod", "Contract"]


def preprocess_for_model(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
    1. Engineer features
    2. One-hot encode categoricals
    3. Return X, y, and the feature list
    """
    df = engineer_features(df)

    # One-hot encode the remaining useful categoricals
    df_encoded = pd.get_dummies(df, columns=["gender", "InternetService"], drop_first=False)

    # Build feature list dynamically (catches whatever get_dummies creates)
    base_features = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "avg_monthly_charge", "charge_ratio", "service_count",
        "contract_risk", "is_senior", "is_electronic_check",
        "has_internet", "paperless", "has_family", "is_high_spender",
    ]
    # Add any one-hot columns from get_dummies
    ohe_features = [c for c in df_encoded.columns
                    if c.startswith("gender_") or c.startswith("InternetService_")]
    feature_cols = base_features + ohe_features

    # Make sure all features exist
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    X = df_encoded[feature_cols].astype(float)
    y = df_encoded["Churn_binary"].astype(int)

    return X, y, feature_cols


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split — stratified because churn is imbalanced."""
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


# --- survival analysis prep ---------------------------------------------------

def prepare_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for Lifelines survival analysis.
    
    In survival analysis:
      - duration = how many months the customer has been with us (tenure)
      - event = 1 if they churned (event observed), 0 if still active (censored)
    
    The censoring idea: a customer who hasn't churned yet doesn't mean they'll
    stay forever — we just don't know their churn time yet.
    """
    df = engineer_features(df)

    survival_df = pd.DataFrame({
        "customer_id": df["customerID"],
        "duration": df["tenure"].clip(lower=1),   # survival duration in months
        "event": df["Churn_binary"],               # 1 = churned (event happened)
        "monthly_charges": df["MonthlyCharges"],
        "total_charges": df["TotalCharges"],
        "contract_risk": df["contract_risk"],
        "service_count": df["service_count"],
        "is_senior": df["is_senior"],
        "has_family": df["has_family"],
        "is_electronic_check": df["is_electronic_check"],
        "has_internet": df["has_internet"],
        "tenure_group": df["tenure_group"],
        "Contract": df["Contract"],
        "gender": df["gender"],
    })

    return survival_df


# --- CLV prep -----------------------------------------------------------------

def compute_clv(df: pd.DataFrame, survival_months: pd.Series,
                monthly_margin_rate: float = 0.30,
                discount_rate: float = 0.10) -> pd.Series:
    """
    Compute a simple but defensible Customer Lifetime Value.
    
    Formula:
        CLV = monthly_charges × margin_rate × expected_remaining_months × discount_factor
    
    Where discount_factor accounts for the time value of money.
    monthly_margin_rate = 30% is a reasonable telecom margin estimate.
    
    This is still a simplification — real CLV models are much more complex!
    But for an internship project, this is rigorous enough.
    """
    monthly_margin = df["MonthlyCharges"] * monthly_margin_rate
    # Rough discount factor: PV of annuity
    r = discount_rate / 12  # monthly discount rate
    n = survival_months.clip(lower=1)
    # PV of annuity formula: (1 - (1+r)^-n) / r
    pv_factor = (1 - (1 + r) ** (-n)) / r
    clv = monthly_margin * pv_factor
    return clv.round(2)
