"""
Feature Engineering & Preprocessing Module
============================================

This module transforms raw telecom records into feature representations optimized
for gradient boosting (LightGBM), survival analysis (Lifelines), and uplift modeling (CausalML).

Core Philosophy:
In tabular machine learning, feature engineering often drives model performance more than
hyperparameter tweaking. Rather than blindly creating mathematical polynomial combinations,
we designed domain-specific features driven by real telecom operational dynamics:

1. Tenure Cohorts (tenure_group):
   Churn hazard is non-linear across time. The first 12 months (the onboarding cliff)
   exhibit drastically higher attrition than established multi-year accounts.
   
2. Billing Dynamics (avg_monthly_charge & charge_ratio):
   Discrepancies between current monthly charges and historical average monthly charges
   often indicate recent price increases or plan modifications, which frequently trigger
   evaluations of competing carriers.
   
3. Ecosystem Stickiness (service_count):
   Customers subscribed to multiple integrated services (security, backups, tech support)
   experience high switching costs, making them significantly more resilient to churn.
   
4. Contractual Commitment (contract_risk):
   Month-to-month contracts have negligible switching friction compared to 1- or 2-year agreements.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Raw Data Ingestion & Sanitization
# -----------------------------------------------------------------------------

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV data and performs initial type sanitization.
    
    Specific sanitization handled:
    - Coerces 'TotalCharges' from string/object to floating-point numeric.
      In the raw IBM Telco dataset, 11 customer records with tenure=0 contain whitespace strings (" ").
    - Imputes those initial nulls with their 'MonthlyCharges' value because they have only had one bill cycle.
    - Encodes the binary target variable 'Churn' (Yes/No) into an integer (1/0).
    """
    df = pd.read_csv(filepath)

    # Convert TotalCharges; spaces turn into NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # For brand-new joiners (tenure == 0), total charges incurred equals their initial monthly charge
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # Binary integer representation for scikit-learn / LightGBM compatibility
    df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)

    return df


# -----------------------------------------------------------------------------
# Domain Feature Engineering
# -----------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the baseline customer attributes with behavioral and financial features.
    
    Each engineered signal maps to a specific behavioral hypothesis:
    """
    df = df.copy()

    # Hypothesis 1: Lifecycle Cohorts
    # Churn rates drop sharply after month 12 and month 24.
    # Grouping into intuitive lifecycle stages helps tree splits and visual analysis.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72, np.inf],
        labels=["0-1 yr", "1-2 yr", "2-4 yr", "4-6 yr", "6+ yr"],
        right=True,
        include_lowest=True
    )

    # Hypothesis 2: Historical vs. Current Monthly Charge Consistency
    # avg_monthly_charge represents the historical burn rate.
    # charge_ratio > 1.0 indicates recent upsell, add-ons, or bill creep.
    df["avg_monthly_charge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["charge_ratio"] = df["MonthlyCharges"] / (df["avg_monthly_charge"] + 1e-6)

    # Hypothesis 3: Product Bundle Stickiness
    # Customers with high service density (OnlineSecurity, TechSupport, etc.)
    # face severe operational pain when migrating to competitors.
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    # Sum only explicit "Yes" selections (ignoring "No phone service" or "No internet service")
    df["service_count"] = df[service_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    # Hypothesis 4: Contract Commitment Friction
    # Month-to-month contracts allow immediate cancellation with zero penalty (highest risk = 3).
    # Two-year contracts lock in pricing and entail early termination penalties (lowest risk = 1).
    contract_risk = {
        "Month-to-month": 3,
        "One year": 2,
        "Two year": 1
    }
    df["contract_risk"] = df["Contract"].map(contract_risk)

    # Senior Citizen flag (re-labeled for clarity)
    df["is_senior"] = df["SeniorCitizen"]

    # Hypothesis 5: Frictionless vs High-Friction Payment Methods
    # EDA demonstrates that Electronic check customers churn at over 40%, whereas
    # automated credit card and bank transfer autopay customers stay significantly longer.
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    # Flag for broadband connectivity
    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    # Digital engagement indicator
    df["paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)

    # Household anchoring: presence of spouse, partner, or dependents provides social stability
    df["has_family"] = (
        (df["Partner"] == "Yes") | (df["Dependents"] == "Yes")
    ).astype(int)

    # Value tier indicator (top quartile monthly spenders)
    df["is_high_spender"] = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)

    return df


# -----------------------------------------------------------------------------
# Model Dataset Preparation
# -----------------------------------------------------------------------------

# Canonical baseline features prioritized for model inputs
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
    "gender_Male",
    "InternetService_Fiber optic",
    "InternetService_No",
]

CATEGORICAL_COLS = ["gender", "InternetService", "PaymentMethod", "Contract"]


def preprocess_for_model(df: pd.DataFrame):
    """
    Executes the full preprocessing pipeline:
    1. Generates engineered features
    2. One-hot encodes remaining categorical attributes
    3. Re-indexes and aligns feature matrices
    
    Returns:
        X (pd.DataFrame): Clean numerical feature matrix.
        y (pd.Series): Binary churn target vector (1=Churn, 0=Stay).
        feature_cols (list): Verified list of input feature column names.
    """
    df = engineer_features(df)

    # One-hot encode categoricals that provide non-ordinal signals
    df_encoded = pd.get_dummies(df, columns=["gender", "InternetService"], drop_first=False)

    # Construct the final feature list dynamically
    base_features = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "avg_monthly_charge", "charge_ratio", "service_count",
        "contract_risk", "is_senior", "is_electronic_check",
        "has_internet", "paperless", "has_family", "is_high_spender",
    ]
    ohe_features = [c for c in df_encoded.columns
                    if c.startswith("gender_") or c.startswith("InternetService_")]
    feature_cols = base_features + ohe_features

    # Safety check: ensure every expected column exists
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    X = df_encoded[feature_cols].astype(float)
    y = df_encoded["Churn_binary"].astype(int)

    return X, y, feature_cols


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Performs stratified train/test splitting.
    
    Why Stratification is Critical:
    The overall churn rate is approximately 26.5% (class imbalance). Stratification guarantees
    that both training and testing partitions retain the exact same 26.5% proportion of churners,
    preventing sampling bias during evaluation.
    """
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


# -----------------------------------------------------------------------------
# Survival Analysis Dataset Preparation
# -----------------------------------------------------------------------------

def prepare_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the customer dataset specifically for Lifelines survival estimators.
    
    Survival Analysis Concepts:
    - Duration: Tenure in months that a customer has continuously stayed.
    - Event (Observed vs Censored):
        * Event = 1: Customer churned during observation window.
        * Event = 0 (Right-Censored): Customer is still active today. We know they survived
          at least 'tenure' months, but their ultimate departure moment is unknown.
          
    Survival analysis models this right-censoring mathematically, avoiding the survivorship
    bias common in standard regression models.
    """
    df = engineer_features(df)

    survival_df = pd.DataFrame({
        "customer_id": df["customerID"],
        "duration": df["tenure"].clip(lower=1),   # Lifelines requires durations >= 1
        "event": df["Churn_binary"],               # 1 = Churned, 0 = Active/Censored
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


# -----------------------------------------------------------------------------
# Customer Lifetime Value (CLV) Calculation
# -----------------------------------------------------------------------------

def compute_clv(df: pd.DataFrame, survival_months: pd.Series,
                monthly_margin_rate: float = 0.30,
                discount_rate: float = 0.10) -> pd.Series:
    """
    Computes Customer Lifetime Value using Discounted Cash Flow (DCF) annuity principles.
    
    Formula:
        CLV = MonthlyCharges * MarginRate * [(1 - (1 + r)^(-n)) / r]
        
    Where:
        - MonthlyCharges * MarginRate = monthly operating contribution margin
        - n = predicted remaining lifetime in months (from the Cox survival model)
        - r = monthly cost of capital / discount rate (annual discount rate / 12)
        - PV Annuity Factor accounts for the time-value of money.
    """
    monthly_margin = df["MonthlyCharges"] * monthly_margin_rate
    r = discount_rate / 12  # monthly discount rate
    n = survival_months.clip(lower=1)
    
    # Present Value of an annuity factor
    pv_factor = (1 - (1 + r) ** (-n)) / r
    clv = monthly_margin * pv_factor
    return clv.round(2)
