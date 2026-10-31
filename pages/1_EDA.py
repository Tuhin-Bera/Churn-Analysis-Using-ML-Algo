"""
Page 1: Exploratory Data Analysis
=====================================

Exploratory Data Analysis (EDA) is the foundational phase of our machine learning lifecycle.
Before training predictive models, we must uncover the underlying structural relationships,
unbalanced distributions, and potential data quality pitfalls across our 7,043 customers.

Key Analytical Focus Areas on this Page:
1. Target Distribution:
   Evaluates class imbalance between churners (26.5%) and retained customers (73.5%).
2. Contractual Attrition Patterns:
   Visualizes the dramatic variation in churn rates across Month-to-month, One-year, and Two-year contracts.
3. Tenure Kinetics:
   Identifies the "early churn cliff" where accounts under 12 months exhibit peak attrition hazard.
4. Multicollinearity & Correlation Analysis:
   Examines linear interactions between tenure, monthly fees, total fees, and model-derived lifetime value.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is available on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="EDA | Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# Custom header styling
st.markdown("""
<style>
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #667eea;
        border-bottom: 2px solid #667eea; padding-bottom: 6px; margin: 20px 0 14px;
    }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #2a2f45);
        border-radius: 12px; padding: 12px 16px; border: 1px solid #3d4460;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Exploratory Data Analysis")
st.caption("Investigating underlying patterns, tenure dynamics, and attrition signals across 7,043 telecom customers.")


# -----------------------------------------------------------------------------
# Data Loading & Caching
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():
    """Reads scored customer observations from the output directory."""
    path = os.path.join(ROOT, "outputs", "master_predictions.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

df = load_data()
if df is None:
    st.error("Outputs missing. Please run `python train_pipeline.py` first.")
    st.stop()


# -----------------------------------------------------------------------------
# High-Level Dataset Geometry
# -----------------------------------------------------------------------------

st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customer Records", f"{len(df):,}")
col2.metric("Original Features", "20")
col3.metric("Engineered Domain Features", "13")
col4.metric("Observed Churn Rate", f"{(df['Churn']=='Yes').mean():.1%}")

with st.expander("Preview Raw Baseline Records (First 50 Rows)"):
    raw_cols = ["customerID","gender","SeniorCitizen","tenure","MonthlyCharges",
                "TotalCharges","Contract","Churn","customer_segment","churn_probability"]
    available = [c for c in raw_cols if c in df.columns]
    st.dataframe(df[available].head(50), use_container_width=True)

st.markdown("---")


# -----------------------------------------------------------------------------
# Target Variable Breakdown: Churn vs. Retention
# -----------------------------------------------------------------------------

st.markdown('<div class="section-header">Target Distribution & Contract Dynamics</div>', unsafe_allow_html=True)
left, right = st.columns([1, 2])

with left:
    # Donut chart of churn proportion
    churn_counts = df["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Status", "Count"]
    pie = alt.Chart(churn_counts).mark_arc(innerRadius=60).encode(
        theta=alt.Theta("Count:Q"),
        color=alt.Color("Status:N",
            scale=alt.Scale(domain=["No","Yes"], range=["#48bb78","#fc8181"])),
        tooltip=["Status","Count"]
    ).properties(height=260, title="Churn Proportions (No vs. Yes)")
    st.altair_chart(pie, use_container_width=True)

with right:
    # Contract type churn comparison
    # Demonstrates that Month-to-month contracts account for the overwhelming majority of churners
    contract_churn = df.groupby("Contract")["Churn"].apply(
        lambda x: (x=="Yes").mean()
    ).reset_index()
    contract_churn.columns = ["Contract", "Churn Rate"]
    bar = alt.Chart(contract_churn).mark_bar().encode(
        x=alt.X("Contract:N", sort="-y", title="Contract Type"),
        y=alt.Y("Churn Rate:Q", axis=alt.Axis(format="%"), title="Observed Churn Percentage"),
        color=alt.Color("Contract:N", scale=alt.Scale(scheme="blues")),
        tooltip=[alt.Tooltip("Churn Rate:Q", format=".1%")]
    ).properties(height=260, title="Churn Rate by Contract Agreement")
    st.altair_chart(bar, use_container_width=True)


# -----------------------------------------------------------------------------
# Tenure Kinetics & Spending Scatter
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Tenure Kinetics & Spending Patterns</div>', unsafe_allow_html=True)
st.markdown("""
> **Key Finding:** Tenure is the single strongest structural anchor against customer churn.
> New joiners in their first year represent over 47% attrition rates, whereas accounts older than 4 years churn at under 8%.
""")

left2, right2 = st.columns(2)

with left2:
    # Histogram of tenure stratified by churn status
    tenure_chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
        x=alt.X("tenure:Q", bin=alt.Bin(step=5), title="Tenure in Months"),
        y=alt.Y("count()", title="Number of Accounts"),
        color=alt.Color("Churn:N",
            scale=alt.Scale(domain=["No","Yes"], range=["#4299e1","#fc8181"])),
        tooltip=["Churn:N", "count()"]
    ).properties(height=280, title="Tenure Distribution by Churn Status")
    st.altair_chart(tenure_chart, use_container_width=True)

with right2:
    # Bivariate scatter: Tenure vs Monthly Charges
    charge_chart = alt.Chart(df.sample(min(1500, len(df)), random_state=42)).mark_circle(
        size=30, opacity=0.5
    ).encode(
        x=alt.X("tenure:Q", title="Tenure (Months)"),
        y=alt.Y("MonthlyCharges:Q", title="Monthly Charges (Rs.)"),
        color=alt.Color("Churn:N",
            scale=alt.Scale(domain=["No","Yes"], range=["#48bb78","#fc8181"])),
        tooltip=["customerID","tenure","MonthlyCharges","Churn"]
    ).properties(height=280, title="Tenure vs. Monthly Charges (Sampled 1,500 Accounts)")
    st.altair_chart(charge_chart, use_container_width=True)


# -----------------------------------------------------------------------------
# Interactive Bivariate Categorical Explorer
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Service & Demographic Categorical Explorer</div>', unsafe_allow_html=True)

cat_col = st.selectbox(
    "Select an operational feature to evaluate its impact on churn:",
    ["Contract", "InternetService", "PaymentMethod", "gender",
     "SeniorCitizen", "Partner", "Dependents", "PhoneService",
     "OnlineSecurity", "TechSupport", "StreamingTV"]
)

# Calculate conditional churn probability for the selected dimension
cat_churn = df.groupby(cat_col)["Churn"].apply(
    lambda x: (x=="Yes").mean()
).reset_index()
cat_churn.columns = [cat_col, "Churn Rate"]
cat_churn = cat_churn.sort_values("Churn Rate", ascending=False)

cat_bar = alt.Chart(cat_churn).mark_bar(color="#667eea").encode(
    x=alt.X(f"{cat_col}:N", sort="-y", title=cat_col),
    y=alt.Y("Churn Rate:Q", axis=alt.Axis(format="%"), title="Conditional Churn Rate"),
    tooltip=[alt.Tooltip("Churn Rate:Q", format=".1%")]
).properties(height=300, title=f"Conditional Churn Rate Grouped by {cat_col}")
st.altair_chart(cat_bar, use_container_width=True)


# -----------------------------------------------------------------------------
# Correlation Heatmap
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Correlation Matrix of Core Numerical Signals</div>', unsafe_allow_html=True)

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                "churn_probability", "clv_estimate", "survival_months"]
numeric_cols = [c for c in numeric_cols if c in df.columns]

corr_df = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, square=True,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9})
ax.set_title("Pearson Correlation Matrix — Key Continuous Metrics", pad=15)
plt.tight_layout()
st.pyplot(fig)
plt.close()


# -----------------------------------------------------------------------------
# Numerical Distribution Boxplots
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Continuous Feature Distribution Explorer</div>', unsafe_allow_html=True)

num_feat = st.selectbox(
    "Select a numerical signal to inspect interquartile range across churn classes:",
    ["tenure", "MonthlyCharges", "TotalCharges",
     "churn_probability", "clv_estimate", "survival_months"]
)

box_chart = alt.Chart(df).mark_boxplot(extent="min-max").encode(
    x=alt.X("Churn:N", title="Churn Status"),
    y=alt.Y(f"{num_feat}:Q", title=num_feat),
    color=alt.Color("Churn:N",
        scale=alt.Scale(domain=["No","Yes"], range=["#48bb78","#fc8181"]))
).properties(height=300, title=f"Distribution of {num_feat} by Attrition Outcome")

st.altair_chart(box_chart, use_container_width=True)


# -----------------------------------------------------------------------------
# Strategic Takeaways
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Synthesis of Key EDA Discoveries</div>', unsafe_allow_html=True)

ins1, ins2, ins3 = st.columns(3)
with ins1:
    st.info("""
    **The Onboarding Cliff (<12 Months)**
    
    Subscribers in their first 12 months churn at an alarming ~47%, whereas those
    exceeding 48 months drop below ~8%. Improving initial customer onboarding is
    the highest-leverage retention strategy.
    """)
with ins2:
    st.warning("""
    **Contract Type Dictates Stability**
    
    Month-to-month contracts exhibit ~43% attrition versus ~11% for 1-year and ~3%
    for 2-year contracts. Incentivizing annual agreement renewals eliminates over
    80% of customer churn risk.
    """)
with ins3:
    st.error("""
    **Payment & Infrastructure Risk Factors**
    
    Electronic check payers (~45% churn) and Fiber Optic users (~41% churn) display
    elevated attrition rates, pointing to potential service dissatisfaction or
    unpleasant manual payment billing cycles.
    """)
