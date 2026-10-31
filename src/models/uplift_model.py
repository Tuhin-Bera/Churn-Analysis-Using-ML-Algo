"""
Uplift Modeling Module — CausalML
===================================

This module estimates Individual Treatment Effects (ITE) and models incremental customer response
using CausalML meta-learners (S-Learner with LightGBM base regressor, and T-Learner fallback).

Why Uplift Modeling is Superior to Standard Propensity Modeling:
Standard churn classifiers rank customers strictly by their probability of leaving:
    P(Churn = 1 | X)

However, offering discounts or incentives solely based on churn probability introduces massive inefficiencies:
1. Sure Things: Loyal customers with low churn risk who would stay regardless. Offering them a discount
   erodes margin without driving incremental retention.
2. Lost Causes: High-risk customers who have already decided to leave due to moving, service outages,
   or severe price sensitivity. Discounts will not change their decision, wasting marketing budget.
3. Do Not Disturb (Sleeping Dogs): Customers whose churn risk might actually increase if contacted
   (e.g., reminding them of an unused subscription or contract renewal).
4. Persuadables: Customers who WILL churn without an incentive, but WILL STAY if an offer is extended.
   THIS is the only cohort that generates positive return on retention investment!

Uplift Formula:
    ITE = E[Y | X, Treatment = 1] - E[Y | X, Treatment = 0]
    
Where:
    - Treatment = 1: Received retention discount offer
    - Treatment = 0: Standard organic experience (control)
    - Y = Outcome (1 if churned, 0 if retained)

Data Limitation Note:
True causal uplift modeling requires randomized A/B trial data (RCT). Because observational datasets
like IBM Telco do not feature randomized offer assignments, we simulate a realistic business assignment rule:
medium-to-high risk accounts receive offers with higher probability, and we model the resulting causal lift.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Treatment Assignment Simulation
# -----------------------------------------------------------------------------

def simulate_treatment_assignment(churn_probs: pd.Series,
                                  seed: int = 42) -> pd.Series:
    """
    Simulates historical retention offer assignment (Treatment = 1 vs 0).
    
    Business Logic:
    In actual enterprise environments, marketing teams rarely discount ultra-loyal users
    (churn_prob < 30%) and often deprioritize accounts deemed completely unsalvageable.
    Therefore, medium-to-high risk customers (30% to 80% churn probability) receive the highest
    treatment probability (~60%), mirroring typical CRM business rules.
    """
    rng = np.random.default_rng(seed)
    
    treatment_prob = np.where(
        (churn_probs >= 0.3) & (churn_probs <= 0.8), 0.6,   # Primary retention target pool
        np.where(churn_probs < 0.3, 0.2,                     # Incidental / test control pool
                 0.4)                                         # High risk salvage pool
    )
    treatment = rng.binomial(1, treatment_prob)
    return pd.Series(treatment, name="treatment")


def simulate_outcome_with_treatment(churn_binary: pd.Series,
                                    churn_probs: pd.Series,
                                    treatment: pd.Series,
                                    lift_effect: float = 0.20,
                                    seed: int = 42) -> pd.Series:
    """
    Simulates counterfactual post-treatment outcomes:
    Did the customer stay after receiving the retention incentive?
    
    Mechanism:
    For treated customers who would otherwise have churned (churn_binary == 1),
    we apply a 20% incremental lift effect (i.e., 20% of at-risk customers change
    their decision and stay active due to the discount).
    """
    rng = np.random.default_rng(seed)
    outcome = churn_binary.copy().values.astype(int)

    # Filter for treated accounts at churn risk
    at_risk_treated = (treatment == 1) & (churn_binary == 1)
    n_affected = at_risk_treated.sum()

    # Simulate true responders
    responders = rng.binomial(1, lift_effect, size=n_affected).astype(bool)
    idx = np.where(at_risk_treated)[0]
    outcome[idx[responders]] = 0   # Saved by intervention!

    return pd.Series(outcome, name="outcome_churn")


# -----------------------------------------------------------------------------
# Uplift Meta-Learners (S-Learner & T-Learner)
# -----------------------------------------------------------------------------

def compute_uplift_scores(X: pd.DataFrame,
                          treatment: pd.Series,
                          outcome: pd.Series) -> pd.Series:
    """
    Computes Individual Treatment Effects (ITE) via a CausalML S-Learner.
    
    S-Learner Architecture:
    Trains a single unified regression model f(X, W) where W is the binary treatment indicator.
    The estimated uplift for any customer is given by:
        Uplift(X) = f(X, W=1) - f(X, W=0)
        
    If CausalML is not installed or encounters environment issues, gracefully falls back
    to a two-model T-Learner using scikit-learn Gradient Boosting.
    """
    try:
        from causalml.inference.meta import BaseSRegressor
        import lightgbm as lgb

        # S-Learner with LightGBM base regressor
        learner = BaseSRegressor(
            learner=lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1
            ),
            control_name=0
        )

        X_with_t = X.copy()
        learner.fit(X_with_t, treatment.values, outcome.values)
        uplift = learner.predict(X_with_t, treatment=treatment.values)

        if hasattr(uplift, "flatten"):
            uplift = uplift.flatten()

        print(f"[Uplift] CausalML S-Learner completed. Score range: [{uplift.min():.3f}, {uplift.max():.3f}]")
        return pd.Series(uplift, name="uplift_score")

    except Exception as e:
        print(f"[Uplift] CausalML S-Learner fallback engaged ({e}). Running T-Learner.")
        return _manual_uplift(X, treatment, outcome)


def _manual_uplift(X: pd.DataFrame,
                   treatment: pd.Series,
                   outcome: pd.Series) -> pd.Series:
    """
    T-Learner (Two-Learner) implementation.
    
    Fits two independent models:
        mu_1(X) = E[Y | X, Treatment = 1]
        mu_0(X) = E[Y | X, Treatment = 0]
        
    ITE is estimated as:
        Uplift(X) = mu_0(X) - mu_1(X)
        (Positive values indicate the discount reduces churn probability).
    """
    from sklearn.ensemble import GradientBoostingClassifier

    X_t = X[treatment == 1]
    y_t = outcome[treatment == 1]
    X_c = X[treatment == 0]
    y_c = outcome[treatment == 0]

    model_t = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model_c = GradientBoostingClassifier(n_estimators=100, random_state=42)

    model_t.fit(X_t, y_t)
    model_c.fit(X_c, y_c)

    # Positive score = intervention saves the customer
    p_treated = model_t.predict_proba(X)[:, 1]
    p_control = model_c.predict_proba(X)[:, 1]
    ite = p_control - p_treated

    print(f"[Uplift] T-Learner completed. Score range: [{ite.min():.3f}, {ite.max():.3f}]")
    return pd.Series(ite, name="uplift_score")


# -----------------------------------------------------------------------------
# Customer Quadrant Categorization & Visualization
# -----------------------------------------------------------------------------

def categorize_uplift(uplift_scores: pd.Series,
                      churn_probs: pd.Series) -> pd.Series:
    """
    Maps each customer into the classical 4-quadrant uplift decision matrix:
    
    1. Persuadable: High Churn Probability (>0.40) & Positive Uplift (>0.05).
       -> Primary targets for promotional discounts.
    2. Lost Cause: High Churn Probability (>0.40) & Low Uplift (<=0.05).
       -> Skip discount offers; cannot be persuaded.
    3. Sure Thing: Low Churn Probability (<=0.40) & Positive Uplift (>0.05).
       -> Skip discount offers; already satisfied.
    4. Do Not Disturb: Low Churn Probability (<=0.40) & Non-positive Uplift (<=0.05).
       -> Do not disturb with marketing pushes.
    """
    categories = []
    for u, c in zip(uplift_scores, churn_probs):
        if u > 0.05 and c > 0.4:
            categories.append("Persuadable")
        elif u <= 0.05 and c > 0.4:
            categories.append("Lost Cause")
        elif u > 0.05 and c <= 0.4:
            categories.append("Sure Thing")
        else:
            categories.append("Do Not Disturb")
    return pd.Series(categories, name="uplift_category")


def plot_uplift_distribution(uplift_scores: pd.Series,
                             categories: pd.Series, save: bool = True):
    """
    Visualizes uplift score distribution and frequency breakdown across the 4 quadrants.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Distribution histogram
    axes[0].hist(uplift_scores, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", label="Zero Uplift Baseline")
    axes[0].set_title("Distribution of Uplift Scores (ITE)", fontsize=12)
    axes[0].set_xlabel("Estimated Individual Treatment Effect")
    axes[0].set_ylabel("Customer Count")
    axes[0].legend()

    # Category breakdown bar chart
    cat_counts = categories.value_counts()
    colors_map = {
        "Persuadable": "green",
        "Lost Cause": "crimson",
        "Sure Thing": "gold",
        "Do Not Disturb": "gray"
    }
    bar_colors = [colors_map.get(c, "steelblue") for c in cat_counts.index]
    cat_counts.plot(kind="bar", ax=axes[1], color=bar_colors, edgecolor="white")
    axes[1].set_title("Customer Segmentation by Uplift Quadrant", fontsize=12)
    axes[1].set_xlabel("Uplift Quadrant")
    axes[1].set_ylabel("Customer Count")
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle("CausalML Uplift Modeling Results", fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "uplift_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Uplift] Saved distribution plot -> {path}")
    plt.close()
