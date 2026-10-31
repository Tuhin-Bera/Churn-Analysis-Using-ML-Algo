"""
Survival Analysis Module — Lifelines (Kaplan-Meier + Cox PH)
=============================================================

This module performs time-to-event survival modeling using the Lifelines package:
1. Kaplan-Meier (KM) Estimator: Non-parametric survival curve estimation
2. Log-Rank Statistical Tests: Significance testing across contract cohorts
3. Cox Proportional Hazards (CPH) Regression: Semi-parametric modeling of baseline hazards
   and feature hazard ratios (e.g., month-to-month contracts vs multi-year agreements)
4. Continuous Survival Forecasting: Estimates remaining median customer lifetime in months

Why Survival Analysis is Necessary:
Standard binary classification (LightGBM) answers only "WILL they leave?".
It cannot answer "WHEN will they leave?". A customer who churns in month 2 represents
a catastrophic acquisition loss; a customer who churns in month 60 has yielded substantial profit.
Furthermore, ordinary linear regression on churned customers suffers from severe survivorship bias
because active accounts (which are right-censored) are ignored or distorted.
Survival analysis treats active accounts as valid censored observations.
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


# -----------------------------------------------------------------------------
# Kaplan-Meier Non-Parametric Survival Analysis
# -----------------------------------------------------------------------------

def fit_kaplan_meier(duration: pd.Series, event: pd.Series,
                     label: str = "All Customers") -> KaplanMeierFitter:
    """
    Fits a non-parametric Kaplan-Meier survival estimator.
    
    Formula:
        S(t) = Product_{t_i <= t} (1 - d_i / n_i)
        
    Where:
        - d_i: Number of churn events at time t_i
        - n_i: Number of individuals known to have survived up to time t_i
    """
    kmf = KaplanMeierFitter()
    kmf.fit(duration, event_observed=event, label=label)
    return kmf


def plot_km_overall(survival_df: pd.DataFrame, save: bool = True):
    """
    Computes and plots the overall survival curve across all 7,043 customers,
    including 95% confidence intervals.
    """
    kmf = fit_kaplan_meier(survival_df["duration"], survival_df["event"])

    fig, ax = plt.subplots(figsize=(9, 5))
    kmf.plot_survival_function(ax=ax, ci_show=True, color="steelblue")
    ax.set_title("Kaplan-Meier Survival Curve — Overall Customer Survival", fontsize=13)
    ax.set_xlabel("Tenure in Months Since Onboarding")
    ax.set_ylabel("Probability of Remaining Active")
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
    Generates comparative survival curves stratified by contract type:
    - Month-to-month
    - One year
    - Two year
    
    Demonstrates the stark difference in attrition kinetics between contract types.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Month-to-month": "crimson", "One year": "orange", "Two year": "steelblue"}

    kmf_dict = {}
    for contract_type, color in colors.items():
        mask = survival_df["Contract"] == contract_type
        if mask.sum() == 0:
            continue
        kmf = fit_kaplan_meier(
            survival_df.loc[mask, "duration"],
            survival_df.loc[mask, "event"],
            label=f"{contract_type} (n={mask.sum():,})"
        )
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)
        kmf_dict[contract_type] = kmf

    ax.set_title("Customer Survival Curves by Contract Type (Kaplan-Meier)", fontsize=13)
    ax.set_xlabel("Tenure in Months")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "km_by_contract.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Survival] Saved KM by contract plot -> {path}")
    plt.close()
    return kmf_dict


def plot_km_by_internet(survival_df: pd.DataFrame, save: bool = True):
    """
    Stratifies customer survival curves by internet infrastructure:
    DSL vs. Fiber Optic vs. No Internet Service.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    group_col = "InternetService" if "InternetService" in survival_df.columns else "internet_service"
    if group_col not in survival_df.columns:
        return None

    palette = sns.color_palette("Set2", len(survival_df[group_col].unique()))
    for i, grp in enumerate(survival_df[group_col].unique()):
        mask = survival_df[group_col] == grp
        kmf = fit_kaplan_meier(
            survival_df.loc[mask, "duration"],
            survival_df.loc[mask, "event"],
            label=f"{grp} (n={mask.sum():,})"
        )
        kmf.plot_survival_function(ax=ax, ci_show=False, color=palette[i])

    ax.set_title("Customer Survival Curves by Internet Service Type", fontsize=13)
    ax.set_xlabel("Tenure in Months")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "km_by_internet.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Survival] Saved KM by internet plot -> {path}")
    plt.close()


# -----------------------------------------------------------------------------
# Cox Proportional Hazards Regression
# -----------------------------------------------------------------------------

def fit_cox_model(survival_df: pd.DataFrame) -> CoxPHFitter:
    """
    Fits a semi-parametric Cox Proportional Hazards model.
    
    Mathematical Formulation:
        h(t | X) = h_0(t) * exp(beta_1 * X_1 + ... + beta_k * X_k)
        
    Where:
        - h(t | X): Hazard rate at time t given covariates X
        - h_0(t): Baseline hazard function
        - exp(beta_i): Hazard Ratio (HR) associated with a unit increase in feature X_i
        
    Interpretation of Hazard Ratios:
        - HR > 1.0: Increases the risk of churn (e.g. month-to-month contracts, electronic check)
        - HR < 1.0: Decreases the risk of churn (e.g. high service count, family anchors)
        - HR = 1.0: Neutral effect on churn risk
    """
    cox_features = [
        "duration", "event",
        "monthly_charges", "contract_risk",
        "service_count", "is_senior",
        "has_family", "is_electronic_check",
        "has_internet",
    ]

    # Keep only available features
    cox_features = [c for c in cox_features if c in survival_df.columns]
    cox_df = survival_df[cox_features].dropna()

    # Cox regression requires strictly positive durations
    cox_df = cox_df[cox_df["duration"] > 0]

    # penalizer=0.1 applies Ridge L2 regularization for stability against mild collinearity
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="duration", event_col="event", show_progress=False)

    print("[Survival] Cox PH model fitted successfully.")
    print(cph.summary[["exp(coef)", "p"]].round(4))
    return cph


def plot_cox_coefficients(cph: CoxPHFitter, save: bool = True):
    """Generates a forest plot of hazard ratios with 95% confidence intervals."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cph.plot(ax=ax)
    ax.set_title("Cox Proportional Hazards — Hazard Ratios (95% CI)", fontsize=13)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "cox_coefficients.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Survival] Saved Cox coefficients plot -> {path}")
    plt.close()


def predict_survival_months(cph: CoxPHFitter, df_features: pd.DataFrame,
                             target_survival_prob: float = 0.5) -> pd.Series:
    """
    Forecasts expected remaining lifetime in months for each customer.
    
    Methodology:
    We compute individual survival functions S_i(t) and determine the median survival
    duration (the milestone month at which the customer's survival probability drops to 50%).
    For ultra-loyal accounts whose survival curve does not drop below 50% within the 72-month
    observation horizon, we conservatively impute the maximum observed horizon.
    """
    feature_cols = [c for c in cph.params_.index if c in df_features.columns]
    X = df_features[feature_cols].fillna(0)

    # Output shape: (time_steps, n_customers)
    survival_funcs = cph.predict_survival_function(X)

    median_survival = []
    for col in survival_funcs.columns:
        sf = survival_funcs[col]
        # Identify first time point where probability drops below median threshold
        below_50 = sf[sf <= target_survival_prob]
        if len(below_50) > 0:
            median_survival.append(below_50.index[0])
        else:
            # Conservative lower bound
            median_survival.append(sf.index[-1])

    return pd.Series(median_survival, name="predicted_survival_months")
