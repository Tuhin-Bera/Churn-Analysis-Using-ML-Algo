"""
Customer Segmentation & CLV Module
=====================================

This module executes two interrelated customer portfolio analyses:
1. Customer Lifetime Value (CLV) Modeling:
   Quantifies the present economic value of every customer over their projected tenure.
   Integrates predicted survival durations from the Lifelines Cox PH model with
   telecom gross margin rates and capital discount rates.
   
2. Unsupervised Behavioral Segmentation (K-Means Clustering):
   Groups 7,043 customer accounts into 4 actionable personas based on behavioral,
   risk, and financial dimensions:
   - High-Value Loyal (Core revenue generators)
   - At-Risk Churners (Immediate intervention targets)
   - Low-Value Passive (Cost-effective upsell targets)
   - New Promising (Early onboarding nurturing cohort)

Why Combine CLV with Segmentation:
Segmenting customers purely by revenue ignores churn risk; segmenting purely by churn risk
ignores economic value. By combining predicted CLV with churn probabilities and tenure,
marketing leadership can allocate retention budgets where the expected return is maximized.
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


# -----------------------------------------------------------------------------
# Customer Lifetime Value (CLV) Modeling
# -----------------------------------------------------------------------------

def compute_clv_detailed(df: pd.DataFrame,
                         survival_months: pd.Series,
                         monthly_margin_rate: float = 0.30,
                         annual_discount_rate: float = 0.10) -> pd.DataFrame:
    """
    Computes discounted customer lifetime value (CLV) via finite annuity present valuation.
    
    Financial Formula:
        CLV = MonthlyCharges * MarginRate * [ (1 - (1 + r)^(-N)) / r ]
        
    Parameters:
        - MonthlyCharges: Current monthly recurring revenue (MRR) per customer.
        - MarginRate: Net contribution margin percentage (~30% for telecom operators).
        - N: Estimated remaining customer lifetime in months (from Cox survival analysis).
        - r: Monthly discount rate = annual_discount_rate / 12 (10% annual / 12 = 0.833% per month).
        - The term [ (1 - (1 + r)^(-N)) / r ] is the Present Value Interest Factor of an Annuity (PVIFA).
    """
    monthly_margin = df["MonthlyCharges"] * monthly_margin_rate
    r = annual_discount_rate / 12
    n = survival_months.clip(lower=1)

    # Annuity discount factor
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
    """
    Generates distribution diagnostics for portfolio Customer Lifetime Value:
    - Histogram with mean portfolio benchmark
    - Scatter plot comparing CLV against monthly charges
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # CLV Histogram
    axes[0].hist(clv_df["clv_estimate"], bins=50,
                 color="teal", edgecolor="white", alpha=0.8)
    axes[0].axvline(clv_df["clv_estimate"].mean(), color="red",
                    linestyle="--", label=f"Mean: Rs.{clv_df['clv_estimate'].mean():,.0f}")
    axes[0].set_title("Distribution of Customer Lifetime Value (CLV)", fontsize=12)
    axes[0].set_xlabel("Predicted CLV (Rs.)")
    axes[0].set_ylabel("Customer Count")
    axes[0].legend()

    # CLV vs Monthly Charges Scatter
    axes[1].scatter(clv_df["monthly_charges"], clv_df["clv_estimate"],
                    alpha=0.3, color="darkorange", s=15)
    axes[1].set_title("CLV vs. Monthly Charges", fontsize=12)
    axes[1].set_xlabel("Current Monthly Charges (Rs.)")
    axes[1].set_ylabel("Predicted Lifetime Value (Rs.)")

    plt.suptitle("Customer Lifetime Value Diagnostics", fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "clv_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[CLV] Saved CLV distribution plot -> {path}")
    plt.close()


# -----------------------------------------------------------------------------
# Customer Behavioral Clustering (K-Means)
# -----------------------------------------------------------------------------

def find_optimal_k(X_scaled: np.ndarray, max_k: int = 8) -> int:
    """
    Evaluates cluster compactness and separation across k in [2, max_k] using:
    1. Elbow Method: Measures Within-Cluster Sum of Squares (Inertia)
    2. Silhouette Score: Measures how well-separated clusters are from neighboring clusters
    """
    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Plot elbow curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(k_range, inertias, "bo-")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (WCSS)")

    axes[1].plot(k_range, silhouettes, "rs-")
    axes[1].set_title("Silhouette Scores")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Mean Silhouette Coefficient")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "clustering_elbow.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Segment] Saved clustering elbow plot -> {path}")

    # Optimal k selection
    best_k = k_range[np.argmax(silhouettes)]
    print(f"[Segment] Optimal k={best_k} (Silhouette={max(silhouettes):.3f})")
    return best_k


# Human-interpretable segment names and recommended marketing playbooks
SEGMENT_NAMES = {
    0: "High-Value Loyal",
    1: "At-Risk Churners",
    2: "Low-Value Passive",
    3: "New Promising",
}

SEGMENT_STRATEGIES = {
    "High-Value Loyal": "Protect and reward: VIP loyalty perks, dedicated concierge support, and early access to upgrades.",
    "At-Risk Churners": "Intervene immediately: Proactive contract extension discounts and customer service health checks.",
    "Low-Value Passive": "Upsell efficiently: Low-touch email bundles promoting add-ons (security, storage) to raise ARPU.",
    "New Promising": "Nurture onboarding: Proactive 30/60/90-day check-ins and onboarding tutorials to prevent early attrition.",
}


def segment_customers(df: pd.DataFrame,
                      churn_probs: pd.Series,
                      clv_estimates: pd.Series,
                      n_clusters: int = 4) -> pd.DataFrame:
    """
    Executes K-Means clustering across 6 core multi-dimensional customer features:
    - tenure (loyalty milestone)
    - monthly_charges (spend rate)
    - churn_probability (attrition risk)
    - clv_estimate (projected lifetime profit)
    - service_count (product breadth)
    - contract_risk (contractual friction)
    
    Standardizes inputs with StandardScaler before clustering to prevent scale dominance.
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

    # Standardize all features to zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # Dynamically assign meaningful segment names based on cluster centroids
    # Rank by net economic health: high CLV and low churn probability
    centers_df = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=cluster_features.columns
    )
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
    """
    Generates 2D scatter visualizations showing segment clusters:
    - Tenure vs. Monthly Charges
    - Churn Risk vs. Customer Lifetime Value
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    palette = {
        "High-Value Loyal": "gold",
        "At-Risk Churners": "crimson",
        "Low-Value Passive": "lightblue",
        "New Promising": "mediumseagreen",
    }

    # Plot 1: Tenure vs Monthly Charges
    for seg, color in palette.items():
        mask = segment_df["segment"] == seg
        if mask.sum() == 0:
            continue
        axes[0].scatter(
            segment_df.loc[mask, "tenure"],
            segment_df.loc[mask, "monthly_charges"],
            c=color, label=seg, alpha=0.5, s=20
        )
    axes[0].set_title("Customer Segments: Tenure vs. Monthly Charges", fontsize=11)
    axes[0].set_xlabel("Tenure (Months)")
    axes[0].set_ylabel("Monthly Charges (Rs.)")
    axes[0].legend(fontsize=8)

    # Plot 2: Churn Probability vs CLV
    for seg, color in palette.items():
        mask = segment_df["segment"] == seg
        if mask.sum() == 0:
            continue
        axes[1].scatter(
            segment_df.loc[mask, "churn_probability"],
            segment_df.loc[mask, "clv_estimate"],
            c=color, label=seg, alpha=0.5, s=20
        )
    axes[1].set_title("Customer Segments: Churn Risk vs. CLV", fontsize=11)
    axes[1].set_xlabel("Churn Probability")
    axes[1].set_ylabel("CLV Estimate (Rs.)")
    axes[1].legend(fontsize=8)

    plt.suptitle("K-Means Customer Segmentation (k=4)", fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "customer_segments.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Segment] Saved segment scatter plot -> {path}")
    plt.close()


def get_segment_summary(segment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates operational summary metrics per customer segment for the Streamlit app.
    """
    summary = segment_df.groupby("segment").agg(
        customer_count=("customer_id", "count"),
        avg_tenure=("tenure", "mean"),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_churn_prob=("churn_probability", "mean"),
        avg_clv=("clv_estimate", "mean"),
    ).round(2)
    summary["strategy"] = summary.index.map(SEGMENT_STRATEGIES)
    return summary
