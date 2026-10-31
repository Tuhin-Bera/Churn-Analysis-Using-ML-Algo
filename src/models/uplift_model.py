"""
Uplift Modeling Module — CausalML
===================================
Author: Tuhin Bera | Data Science Intern
Date: October 2026

Uplift modeling answers a subtler question than churn prediction:
"Which customers will stay BECAUSE of our offer, not despite it?"

There are 4 types of customers in any retention campaign:
  1. Persuadables  — would churn without offer, stay WITH offer  ← WE WANT THESE
  2. Sure Things   — would stay regardless (wasting money on them)
  3. Lost Causes   — will churn regardless of offer
  4. Do Not Disturb — might churn IF offered (rare, but real)

Since we don't have a real randomized experiment (treatment/control assignment),
I'm using a meta-learner approach (S-Learner, T-Learner) and simulating
treatment assignment based on a plausible business rule.

This is academically honest — I explicitly document the limitation.

NOTE ON DATA LIMITATION:
  True uplift modeling requires randomized control trial data.
  The IBM Telco dataset doesn't have this. So I simulate treatment assignment
  using a rule that mimics what a real business might do:
    - Customers at medium-high churn risk (30-70%) get offered a discount
    - Others don't
  This is a simplified but educational demonstration.
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


def simulate_treatment_assignment(churn_probs: pd.Series,
                                  seed: int = 42) -> pd.Series:
    """
    Simulate which customers received a retention offer (treatment=1).
    
    Business logic: medium-risk customers (30-70% churn probability)
    are most likely to receive offers in practice. Very high-risk and
    very low-risk customers are often excluded from discount campaigns.
    
    Returns a binary Series: 1 = received offer, 0 = did not receive offer
    """
    rng = np.random.default_rng(seed)
    # Base probability of receiving treatment (offer)
    treatment_prob = np.where(
        (churn_probs >= 0.3) & (churn_probs <= 0.8), 0.6,   # medium risk
        np.where(churn_probs < 0.3, 0.2,                     # low risk
                 0.4)                                         # very high risk
    )
    treatment = rng.binomial(1, treatment_prob)
    return pd.Series(treatment, name="treatment")


def simulate_outcome_with_treatment(churn_binary: pd.Series,
                                    churn_probs: pd.Series,
                                    treatment: pd.Series,
                                    lift_effect: float = 0.20,
                                    seed: int = 42) -> pd.Series:
    """
    Simulate a post-treatment outcome (did they churn after receiving offer?).
    
    For treated customers who were going to churn (churn=1),
    there's a `lift_effect` probability that they changed their mind.
    
    This is the synthetic outcome we'll use to train the uplift model.
    """
    rng = np.random.default_rng(seed)
    outcome = churn_binary.copy().values.astype(int)

    # For treated customers who originally churn: apply lift
    at_risk_treated = (treatment == 1) & (churn_binary == 1)
    n_affected = at_risk_treated.sum()

    # Some percentage of these respond to the offer and don't churn
    responders = rng.binomial(1, lift_effect, size=n_affected).astype(bool)
    idx = np.where(at_risk_treated)[0]
    outcome[idx[responders]] = 0   # they stayed!

    return pd.Series(outcome, name="outcome_churn")


def compute_uplift_scores(X: pd.DataFrame,
                          treatment: pd.Series,
                          outcome: pd.Series) -> pd.Series:
    """
    S-Learner uplift model using LightGBM.
    
    S-Learner trains ONE model with treatment as a feature.
    Uplift = P(outcome | T=1) - P(outcome | T=0)
    
    This gives us the estimated individual treatment effect (ITE).
    Positive uplift -> customer is likely to respond to offer.
    """
    try:
        from causalml.inference.meta import BaseSRegressor
        import lightgbm as lgb

        # S-learner with LightGBM base
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

        # CausalML expects (X, treatment, y) — note treatment is the variable name
        X_with_t = X.copy()
        learner.fit(X_with_t, treatment.values, outcome.values)
        uplift = learner.predict(X_with_t, treatment=treatment.values)

        # uplift is typically returned as ndarray
        if hasattr(uplift, "flatten"):
            uplift = uplift.flatten()

        print(f"[Uplift] S-Learner computed. Uplift range: [{uplift.min():.3f}, {uplift.max():.3f}]")
        return pd.Series(uplift, name="uplift_score")

    except Exception as e:
        print(f"[Uplift] CausalML S-Learner failed ({e}). Falling back to manual computation.")
        return _manual_uplift(X, treatment, outcome)


def _manual_uplift(X: pd.DataFrame,
                   treatment: pd.Series,
                   outcome: pd.Series) -> pd.Series:
    """
    Fallback: T-Learner approach with sklearn.
    Trains two models — one for treated, one for control — then
    computes ITE = P_treatment(churn) - P_control(churn).
    
    We negate because we want high uplift = good (likely to stay with offer).
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

    # ITE = P(churn | treated) - P(churn | control)
    # Negative ITE means treatment REDUCES churn -> good!
    p_treated = model_t.predict_proba(X)[:, 1]
    p_control = model_c.predict_proba(X)[:, 1]
    ite = p_control - p_treated  # positive = offer helps

    print(f"[Uplift] T-Learner fallback. Uplift range: [{ite.min():.3f}, {ite.max():.3f}]")
    return pd.Series(ite, name="uplift_score")


def categorize_uplift(uplift_scores: pd.Series,
                      churn_probs: pd.Series) -> pd.Series:
    """
    Classify customers into the four Qini quadrants.
    
    Persuadables (high uplift + high churn risk) = priority targets
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
    """Histogram of uplift scores colored by category."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: distribution
    axes[0].hist(uplift_scores, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", label="Zero uplift")
    axes[0].set_title("Distribution of Uplift Scores", fontsize=12)
    axes[0].set_xlabel("Uplift Score (ITE)")
    axes[0].set_ylabel("Customer Count")
    axes[0].legend()

    # Right: category counts
    cat_counts = categories.value_counts()
    colors_map = {
        "Persuadable": "green",
        "Lost Cause": "crimson",
        "Sure Thing": "gold",
        "Do Not Disturb": "gray"
    }
    bar_colors = [colors_map.get(c, "steelblue") for c in cat_counts.index]
    cat_counts.plot(kind="bar", ax=axes[1], color=bar_colors, edgecolor="white")
    axes[1].set_title("Customer Segments by Uplift Category", fontsize=12)
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle("Uplift Modeling Results (CausalML S-Learner)", fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "uplift_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Uplift] Saved distribution plot -> {path}")
    plt.close()
