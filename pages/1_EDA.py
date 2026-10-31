"""
Page 1: Exploratory Data Analysis
=====================================

EDA is always the first thing we do before building any model.
We need to understand our data before we can model it.

This page shows:
- Distribution of key features
- Churn rate by different customer segments
- Correlation analysis
- Feature distributions by churn status
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="EDA | Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# ─── CSS (reuse same style) ───────────────────────────────────────────────────
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
st.caption("Understanding our 7,043 telecom customers before building models.")


# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(ROOT, "outputs", "master_predictions.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

df = load_data()
if df is None:
    st.error("Run `python train_pipeline.py` first.")
    st.stop()

# ─── Dataset Overview ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Features (original)", "20")
col3.metric("Engineered Features", "13")
col4.metric("Churn Rate", f"{(df['Churn']=='Yes').mean():.1%}")

with st.expander("Preview Raw Data (first 50 rows)"):
    raw_cols = ["customerID","gender","SeniorCitizen","tenure","MonthlyCharges",
                "TotalCharges","Contract","Churn","customer_segment","churn_probability"]
    available = [c for c in raw_cols if c in df.columns]
    st.dataframe(df[available].head(50), use_container_width=True)

st.markdown("---")

# ─── Churn Distribution ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Churn Distribution</div>', unsafe_allow_html=True)
left, right = st.columns([1, 2])

with left:
    churn_counts = df["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Status", "Count"]
    pie = alt.Chart(churn_counts).mark_arc(innerRadius=60).encode(
        theta=alt.Theta("Count:Q"),
        color=alt.Color("Status:N",
            scale=alt.Scale(domain=["No","Yes"], range=["#48bb78","#fc8181"])),
        tooltip=["Status","Count"]
    ).properties(height=260, title="Churn vs Retention")
    st.altair_chart(pie, use_container_width=True)

with right:
    # Churn rate by contract type
    contract_churn = df.groupby("Contract")["Churn"].apply(
        lambda x: (x=="Yes").mean()
    ).reset_index()
    contract_churn.columns = ["Contract", "Churn Rate"]
    bar = alt.Chart(contract_churn).mark_bar().encode(
        x=alt.X("Contract:N", sort="-y"),
        y=alt.Y("Churn Rate:Q", axis=alt.Axis(format="%")),
        color=alt.Color("Contract:N", scale=alt.Scale(scheme="blues")),
        tooltip=[alt.Tooltip("Churn Rate:Q", format=".1%")]
    ).properties(height=260, title="Churn Rate by Contract Type")
    st.altair_chart(bar, use_container_width=True)

# ─── Tenure Analysis ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Tenure Analysis</div>', unsafe_allow_html=True)
st.markdown("""
> Tenure is one of the strongest predictors of churn.
> New customers (under 12 months) are far more likely to leave.
""")

left2, right2 = st.columns(2)

with left2:
    # Tenure histogram by churn
    tenure_chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
        x=alt.X("tenure:Q", bin=alt.Bin(step=5), title="Tenure (months)"),
        y=alt.Y("count()", title="Customers"),
        color=alt.Color("Churn:N",
            scale=alt.Scale(domain=["No","Yes"], range=["#4299e1","#fc8181"])),
        tooltip=["Churn:N", "count()"]
    ).properties(height=280, title="Tenure Distribution by Churn Status")
    st.altair_chart(tenure_chart, use_container_width=True)

with right2:
    # Monthly charges vs churn
    charge_chart = alt.Chart(df.sample(min(1500, len(df)), random_state=42)).mark_circle(
        size=30, opacity=0.5
    ).encode(
        x=alt.X("tenure:Q", title="Tenure (months)"),
        y=alt.Y("MonthlyCharges:Q", title="Monthly Charges"),
        color=alt.Color("Churn:N",
            scale=alt.Scale(domain=["No","Yes"], range=["#48bb78","#fc8181"])),
        tooltip=["customerID","tenure","MonthlyCharges","Churn"]
    ).properties(height=280, title="Tenure vs Monthly Charges (colored by churn)")
    st.altair_chart(charge_chart, use_container_width=True)

# ─── Service Analysis ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Service & Demographics Analysis</div>', unsafe_allow_html=True)

cat_col = st.selectbox(
    "Select a categorical feature to analyze churn rate:",
    ["Contract", "InternetService", "PaymentMethod", "gender",
     "SeniorCitizen", "Partner", "Dependents", "PhoneService",
     "OnlineSecurity", "TechSupport", "StreamingTV"]
)

cat_churn = df.groupby(cat_col)["Churn"].apply(
    lambda x: (x=="Yes").mean()
).reset_index()
cat_churn.columns = [cat_col, "Churn Rate"]
cat_churn = cat_churn.sort_values("Churn Rate", ascending=False)

cat_bar = alt.Chart(cat_churn).mark_bar(color="#667eea").encode(
    x=alt.X(f"{cat_col}:N", sort="-y"),
    y=alt.Y("Churn Rate:Q", axis=alt.Axis(format="%")),
    tooltip=[alt.Tooltip("Churn Rate:Q", format=".1%")]
).properties(height=300, title=f"Churn Rate by {cat_col}")
st.altair_chart(cat_bar, use_container_width=True)

# ─── Correlation Heatmap ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Correlation Analysis (Numeric Features)</div>', unsafe_allow_html=True)

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                "churn_probability", "clv_estimate", "survival_months"]
numeric_cols = [c for c in numeric_cols if c in df.columns]

corr_df = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, square=True,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9})
ax.set_title("Correlation Matrix — Key Numeric Features", pad=15)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─── Feature Distributions ────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Feature Distribution Explorer</div>', unsafe_allow_html=True)

num_feat = st.selectbox(
    "Select a numeric feature to explore:",
    ["tenure", "MonthlyCharges", "TotalCharges",
     "churn_probability", "clv_estimate", "survival_months"]
)

box_chart = alt.Chart(df).mark_boxplot(extent="min-max").encode(
    x=alt.X("Churn:N"),
    y=alt.Y(f"{num_feat}:Q", title=num_feat),
    color=alt.Color("Churn:N",
        scale=alt.Scale(domain=["No","Yes"], range=["#48bb78","#fc8181"]))
).properties(height=300, title=f"{num_feat} Distribution by Churn Status")

st.altair_chart(box_chart, use_container_width=True)

# ─── Key Insights ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Key EDA Insights</div>', unsafe_allow_html=True)

ins1, ins2, ins3 = st.columns(3)
with ins1:
    st.info("""
    **Short tenure = high churn**
    
    Customers with < 12 months tenure have a churn rate of ~47%.
    Customers with > 48 months tenure have a churn rate of only ~8%.
    Onboarding programs are critical.
    """)
with ins2:
    st.warning("""
    **Month-to-month contracts are risky**
    
    Month-to-month contract customers churn at ~43%.
    One-year: ~11%. Two-year: ~3%.
    Converting customers to longer contracts is the #1 retention lever.
    """)
with ins3:
    st.error("""
    **Electronic check + Fiber optic = highest risk**
    
    These two groups show the highest churn rates.
    Fiber optic churn rate: ~41%.
    Electronic check churn rate: ~45%.
    Both may indicate dissatisfaction with service quality or price.
    """)

