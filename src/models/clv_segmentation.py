"""
Customer Segmentation & CLV Module
=====================================
Author: Tuhin Bera | Data Science Intern
Date: October 2026

Two parts:
1. CLV (Customer Lifetime Value) — how much is each customer worth going forward?
2. K-Means clustering — group customers into actionable segments

I use the survival model's predicted remaining months for CLV estimation,
which makes it more sophisticated than just tenure × monthly_charges.

For clustering, I tried K-Means with k=2,3,4,5 and used elbow method + 
silhouette scores to pick k=4 (it gave the most interpretable segments).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- CLV computation ----------------------------------------------------------

def compute_clv_detailed(df: pd.DataFrame,
                         survival_months: pd.Series,
                         monthly_margin_rate: float = 0.30,
                         annual_discount_rate: float = 0.10) -> pd.DataFrame:
    """
    Compute CLV for each customer using the present value of future cash flows.
    
    CLV = Σ (monthly_margin × discount_factor_t) for t=1..N_months
        = monthly_margin × PV_annuity_factor
    
    Where:
        monthly_margin = MonthlyCharges × margin_rate
        N_months = predicted remaining lifetime from survival model
        PV_annuity_factor = (1 - (1+r)^-N) / r
        r = annual_discount_rate / 12
    
    This is a simplified model. Real CLV would also include:
        - Acquisition costs (already sunk, so often excluded)
        - Churn probability over each period
        - Cross-sell / upsell potential
    """
    monthly_margin = df["MonthlyCharges"] * monthly_margin_rate
    r = annual_discount_rate / 12  # monthly discount rate
    n = survival_months.clip(lower=1)

    # Present value annuity factor
    pv_factor = (1 - (1 + r) ** (-n)) / r

    clv = (monthly_margin.values * pv_factor.values).round(2)

    result = pd.DataFrame({
        "customer_id": df["customerID"],
        "monthly_charges": df["MonthlyCharges"],
        "monthly_margin": monthly_margin.round(2),
        "predicted_months_remaining": n.round(1),
        "clv_estimate": clv,
    })
    return result


def plot_clv_distribution(clv_df: pd.DataFrame, save: bool = True):
    """Distribution of CLV across all customers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram
    axes[0].hist(clv_df["clv_estimate"], bins=50,
                 color="teal", edgecolor="white", alpha=0.8)
    axes[0].axvline(clv_df["clv_estimate"].mean(), color="red",
                    linestyle="--", label=f"Mean: ₹{clv_df['clv_estimate'].mean():,.0f}")
    axes[0].set_title("Distribution of Customer Lifetime Value", fontsize=12)
    axes[0].set_xlabel("CLV (₹)")
    axes[0].set_ylabel("Number of Customers")
    axes[0].legend()

    # Right: CLV vs monthly charges scatter
    axes[1].scatter(clv_df["monthly_charges"], clv_df["clv_estimate"],
                    alpha=0.3, color="darkorange", s=15)
    axes[1].set_title("CLV vs Monthly Charges", fontsize=12)
    axes[1].set_xlabel("Monthly Charges (₹)")
    axes[1].set_ylabel("CLV Estimate (₹)")

    plt.suptitle("Customer Lifetime Value Analysis", fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "clv_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[CLV] Saved CLV distribution -> {path}")
    plt.close()


# --- Customer Segmentation ----------------------------------------------------

def find_optimal_k(X_scaled: np.ndarray, max_k: int = 8) -> int:
    """
    Use elbow method + silhouette score to find optimal number of clusters.
    Returns the k with the highest silhouette score (tiebreak by inertia elbow).
    """
    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Plot both
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(k_range, inertias, "bo-")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(k_range, silhouettes, "rs-")
    axes[1].set_title("Silhouette Scores")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "clustering_elbow.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Segment] Saved elbow plot -> {path}")

    # Best k by silhouette
    best_k = k_range[np.argmax(silhouettes)]
    print(f"[Segment] Optimal k={best_k} (silhouette={max(silhouettes):.3f})")
    return best_k


SEGMENT_NAMES = {
    0: "High-Value Loyal",
    1: "At-Risk Churners",
    2: "Low-Value Passive",
    3: "New Promising",
}

SEGMENT_STRATEGIES = {
    "High-Value Loyal": "Retain with loyalty rewards and premium service upgrades.",
    "At-Risk Churners": "Priority for win-back campaigns; offer targeted discounts.",
    "Low-Value Passive": "Low priority; focus on upselling to increase value.",
    "New Promising": "Nurture with onboarding assistance; prevent early churn.",
}


def segment_customers(df: pd.DataFrame,
                      churn_probs: pd.Series,
                      clv_estimates: pd.Series,
                      n_clusters: int = 4) -> pd.DataFrame:
    """
    K-Means clustering on key customer metrics.
    
    Clustering features chosen for business interpretability:
    - tenure: loyalty indicator
    - monthly_charges: value indicator
    - churn probability: risk indicator
    - CLV: future value indicator
    - service_count: engagement indicator
    """
    from src.utils.feature_engineering import engineer_features
    df_feat = engineer_features(df)

    cluster_features = pd.DataFrame({
        "tenure": df["tenure"],
        "monthly_charges": df["MonthlyCharges"],
        "churn_probability": churn_probs.values,
        "clv_estimate": clv_estimates.values,
        "service_count": df_feat["service_count"].values,
        "contract_risk": df_feat["contract_risk"].values,
    })
    cluster_features = cluster_features.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # Assign meaningful names based on cluster centers
    # We'll sort clusters by (avg CLV desc, avg churn_prob asc)
    centers_df = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=cluster_features.columns
    )
    # Rank by CLV descending and churn ascending
    centers_df["score"] = centers_df["clv_estimate"] - centers_df["churn_probability"] * 100
    rank_order = centers_df["score"].rank(ascending=False).astype(int) - 1

    name_map = {
        original: list(SEGMENT_NAMES.values())[rank]
        for original, rank in rank_order.items()
    }
    segment_labels = [name_map[l] for l in labels]

    result_df = cluster_features.copy()
    result_df["customer_id"] = df["customerID"].values
    result_df["cluster_id"] = labels
    result_df["segment"] = segment_labels

    return result_df, km, scaler, cluster_features.columns.tolist()


def plot_segments(segment_df: pd.DataFrame, save: bool = True):
    """Scatter plot of customer segments: tenure vs monthly charges."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    palette = {
        "High-Value Loyal": "gold",
        "At-Risk Churners": "crimson",
        "Low-Value Passive": "lightblue",
        "New Promising": "mediumseagreen",
    }

    # Tenure vs Monthly Charges
    for seg, color in palette.items():
        mask = segment_df["segment"] == seg
        if mask.sum() == 0:
            continue
        axes[0].scatter(
            segment_df.loc[mask, "tenure"],
            segment_df.loc[mask, "monthly_charges"],
            c=color, label=seg, alpha=0.5, s=20
        )
    axes[0].set_title("Customer Segments: Tenure vs Monthly Charges", fontsize=11)
    axes[0].set_xlabel("Tenure (months)")
    axes[0].set_ylabel("Monthly Charges (₹)")
    axes[0].legend(fontsize=8)

    # CLV vs Churn Probability
    for seg, color in palette.items():
        mask = segment_df["segment"] == seg
        if mask.sum() == 0:
            continue
        axes[1].scatter(
            segment_df.loc[mask, "churn_probability"],
            segment_df.loc[mask, "clv_estimate"],
            c=color, label=seg, alpha=0.5, s=20
        )
    axes[1].set_title("Customer Segments: Churn Risk vs CLV", fontsize=11)
    axes[1].set_xlabel("Churn Probability")
    axes[1].set_ylabel("CLV Estimate (₹)")
    axes[1].legend(fontsize=8)

    plt.suptitle("K-Means Customer Segmentation (k=4)", fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "customer_segments.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Segment] Saved segment scatter -> {path}")
    plt.close()


def get_segment_summary(segment_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statistics per segment for the dashboard."""
    summary = segment_df.groupby("segment").agg(
        customer_count=("customer_id", "count"),
        avg_tenure=("tenure", "mean"),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_churn_prob=("churn_probability", "mean"),
        avg_clv=("clv_estimate", "mean"),
    ).round(2)
    summary["strategy"] = summary.index.map(SEGMENT_STRATEGIES)
    return summary
