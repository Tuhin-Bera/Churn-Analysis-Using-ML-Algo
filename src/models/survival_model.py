"""
Survival Analysis Module — Lifelines (Kaplan-Meier + Cox PH)
=============================================================

Survival analysis answers: "How long will a customer stay?"
- Kaplan-Meier: non-parametric survival curve (no assumptions about distribution)
- Cox Proportional Hazards: semi-parametric model, shows which features
  increase/decrease churn hazard

Key concept: CENSORING
  A customer who hasn't churned yet is "censored" — we know they survived
  at least this long, but we don't know when (or if) they'll churn.
  Survival analysis handles censoring correctly, which is why it's better
  than just doing regression on churned customers only.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Kaplan-Meier -------------------------------------------------------------

def fit_kaplan_meier(duration: pd.Series, event: pd.Series,
                     label: str = "All Customers") -> KaplanMeierFitter:
    """
    Fit a Kaplan-Meier survival estimator.
    
    Args:
        duration: months each customer has been active
        event: 1 if they churned, 0 if still active (censored)
    """
    kmf = KaplanMeierFitter()
    kmf.fit(duration, event_observed=event, label=label)
    return kmf


def plot_km_overall(survival_df: pd.DataFrame, save: bool = True):
    """Overall survival curve for all customers."""
    kmf = fit_kaplan_meier(survival_df["duration"], survival_df["event"])

    fig, ax = plt.subplots(figsize=(9, 5))
    kmf.plot_survival_function(ax=ax, ci_show=True, color="steelblue")
    ax.set_title("Kaplan-Meier Survival Curve — Overall Customer Survival", fontsize=13)
    ax.set_xlabel("Months since joining")
    ax.set_ylabel("Probability of remaining a customer")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "km_overall.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Survival] Saved KM overall plot -> {path}")
    plt.close()
    return kmf


def plot_km_by_contract(survival_df: pd.DataFrame, save: bool = True):
    """
    Compare survival curves by contract type.
    This is one of the most insightful views — monthly contracts churn much faster.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Month-to-month": "crimson", "One year": "orange", "Two year": "steelblue"}

    p_values = {}
    kmf_dict = {}
    for contract_type, color in colors.items():
        mask = survival_df["Contract"] == contract_type
        if mask.sum() == 0:
            continue
        kmf = fit_kaplan_meier(
            survival_df.loc[mask, "duration"],
            survival_df.loc[mask, "event"],
            label=f"{contract_type} (n={mask.sum()})"
        )
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)
        kmf_dict[contract_type] = kmf

    ax.set_title("Survival Curves by Contract Type", fontsize=13)
    ax.set_xlabel("Months since joining")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "km_by_contract.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Survival] Saved KM by contract -> {path}")
    plt.close()
    return kmf_dict


def plot_km_by_internet(survival_df: pd.DataFrame, save: bool = True):
    """Survival by internet service type — fiber optic customers churn more."""
    fig, ax = plt.subplots(figsize=(10, 6))
    internet_types = survival_df["internet_service"].unique() if "internet_service" in survival_df.columns else []
    if "InternetService" in survival_df.columns:
        group_col = "InternetService"
    elif "internet_service" in survival_df.columns:
        group_col = "internet_service"
    else:
        return None

    palette = sns.color_palette("Set2", len(survival_df[group_col].unique()))
    for i, grp in enumerate(survival_df[group_col].unique()):
        mask = survival_df[group_col] == grp
        kmf = fit_kaplan_meier(
            survival_df.loc[mask, "duration"],
            survival_df.loc[mask, "event"],
            label=f"{grp} (n={mask.sum()})"
        )
        kmf.plot_survival_function(ax=ax, ci_show=False, color=palette[i])

    ax.set_title("Survival Curves by Internet Service Type", fontsize=13)
    ax.set_xlabel("Months since joining")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "km_by_internet.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Survival] Saved KM by internet -> {path}")
    plt.close()


# --- Cox Proportional Hazards -------------------------------------------------

def fit_cox_model(survival_df: pd.DataFrame) -> CoxPHFitter:
    """
    Fit a Cox Proportional Hazards model.
    
    Cox model estimates the hazard ratio for each feature.
    hazard ratio > 1 -> increases churn risk
    hazard ratio < 1 -> decreases churn risk
    
    We are careful to only include numeric features and remove multicollinear ones.
    """
    cox_features = [
        "duration", "event",
        "monthly_charges", "contract_risk",
        "service_count", "is_senior",
        "has_family", "is_electronic_check",
        "has_internet",
    ]

    # Keep only columns that exist in the dataframe
    cox_features = [c for c in cox_features if c in survival_df.columns]
    cox_df = survival_df[cox_features].dropna()

    # Cox requires positive durations
    cox_df = cox_df[cox_df["duration"] > 0]

    cph = CoxPHFitter(penalizer=0.1)   # small L2 penalty for stability
    cph.fit(cox_df, duration_col="duration", event_col="event",
            show_progress=False)

    print("[Survival] Cox PH model fitted.")
    print(cph.summary[["exp(coef)", "p"]].round(4))
    return cph


def plot_cox_coefficients(cph: CoxPHFitter, save: bool = True):
    """Forest plot of Cox hazard ratios."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cph.plot(ax=ax)
    ax.set_title("Cox PH Model — Hazard Ratios with 95% CI", fontsize=13)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "cox_coefficients.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Survival] Saved Cox coefficients -> {path}")
    plt.close()


def predict_survival_months(cph: CoxPHFitter, df_features: pd.DataFrame,
                             target_survival_prob: float = 0.5) -> pd.Series:
    """
    Estimate how many more months each customer is expected to survive.
    We use the median survival time (50th percentile) as the estimate.
    
    For customers where the model doesn't reach 50% survival probability
    within the observed window, we use the maximum observed time as a
    conservative estimate.
    """
    # We need the same features used during training
    feature_cols = [c for c in cph.params_.index if c in df_features.columns]
    X = df_features[feature_cols].fillna(0)

    survival_funcs = cph.predict_survival_function(X)
    # survival_funcs is shape (times, n_customers)

    median_survival = []
    for col in survival_funcs.columns:
        sf = survival_funcs[col]
        # Find the time where survival probability first drops below 0.5
        below_50 = sf[sf <= target_survival_prob]
        if len(below_50) > 0:
            median_survival.append(below_50.index[0])
        else:
            # Didn't drop below 50% — use max time as lower bound
            median_survival.append(sf.index[-1])

    return pd.Series(median_survival, name="predicted_survival_months")
