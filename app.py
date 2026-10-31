"""
app.py - Main Streamlit Dashboard Entry Point
==============================================
Customer Lifetime Value & Churn Intelligence System

This is the primary Streamlit entry point. It serves two distinct functions:
1. Executive Overview & Portfolio KPI Summary:
   Provides high-level leadership summaries answering the four core business questions:
   - Who will leave? (LightGBM churn probability)
   - How long will they stay? (Survival analysis remaining months)
   - How much are they worth? (Discounted CLV)
   - How can we retain them? (Uplift modeling target persuasion)
   
2. Multi-Page Navigation Hub:
   Establishes sidebar routing across all deep-dive analytical pages:
   - 1_EDA: Exploratory Data Analysis & Bivariate Distributions
   - 2_Churn_Predictor: Individual Customer Lookups & Local SHAP Waterfall Plots
   - 3_Survival_Analysis: Kaplan-Meier Curves & Cox Proportional Hazards
   - 4_Uplift_Modeling: CausalML 4-Quadrant Targeting (Persuadables vs. Lost Causes)
   - 5_CLV_Segments: Present Value CLV Distributions & 4-Cluster Profiles
   - 6_ROI_Simulator: Dynamic Campaign Budget & Retention Offer Optimizer

Execution:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# Ensure the root project directory is discoverable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Global Theming
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Churn Intelligence System | Nexvia",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for an executive dark-mode user interface
st.markdown("""
<style>
    /* Dark background canvas */
    .main { background-color: #0e1117; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    /* Elevated KPI metric containers */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #2a2f45);
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #3d4460;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] label {
        color: #a0aec0 !important;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* Hero header banner with gradient */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 30px 40px;
        margin-bottom: 24px;
        color: white;
    }
    .hero-banner h1 { font-size: 2.2rem; margin: 0; font-weight: 700; }
    .hero-banner p { font-size: 1.05rem; margin: 8px 0 0; opacity: 0.88; }

    /* Strategic question cards */
    .info-card {
        background: #1a1f35;
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .info-card h4 { color: #667eea; margin: 0 0 8px; }
    .info-card p { color: #cbd5e0; margin: 0; font-size: 0.9rem; }

    /* Sidebar navigation styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f35 0%, #0e1117 100%);
        border-right: 1px solid #2d3748;
    }
    [data-testid="stSidebar"] .element-container { color: #e2e8f0; }

    /* Uniform section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #667eea;
        border-bottom: 2px solid #667eea;
        padding-bottom: 6px;
        margin: 20px 0 14px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# High-Performance Data & Model Caching
# -----------------------------------------------------------------------------

@st.cache_data
def load_master_data():
    """
    Loads pre-computed customer predictions from the outputs directory.
    Using @st.cache_data ensures the 7,043-row dataset is cached in memory,
    preventing disk read delays during user interactions.
    """
    path = os.path.join(ROOT, "outputs", "master_predictions.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


@st.cache_resource
def load_model_artifacts():
    """
    Loads the serialized LightGBM model and SHAP TreeExplainer object.
    Using @st.cache_resource guarantees model weights are only unpickled once.
    """
    path = os.path.join(ROOT, "outputs", "lgbm_churn_model.pkl")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["explainer"]


@st.cache_data
def load_meta():
    """Loads pipeline metadata (feature names, test set metrics, segment mappings)."""
    path = os.path.join(ROOT, "outputs", "pipeline_meta.pkl")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Sidebar Navigation & Track Information
# -----------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/signal-tower.png", width=60)
    st.markdown("## Churn Intelligence")
    st.markdown("*Nexvia Technologies | DS-01*")
    st.markdown("---")

    # Seamless page routing links
    st.markdown("### Navigation")
    st.page_link("app.py",              label="🏠 Overview & KPIs",        icon=None)
    st.page_link("pages/1_EDA.py",      label="📊 Exploratory Analysis",    icon=None)
    st.page_link("pages/2_Churn_Predictor.py", label="🔮 Churn Predictor",  icon=None)
    st.page_link("pages/3_Survival_Analysis.py", label="⏳ Survival Analysis", icon=None)
    st.page_link("pages/4_Uplift_Modeling.py", label="🎯 Uplift Modeling",  icon=None)
    st.page_link("pages/5_CLV_Segments.py", label="💰 CLV & Segments",      icon=None)
    st.page_link("pages/6_ROI_Simulator.py", label="📈 ROI Simulator",      icon=None)

    st.markdown("---")
    st.markdown("**Track:** Data Science")


# -----------------------------------------------------------------------------
# Executive Landing Dashboard
# -----------------------------------------------------------------------------

# Primary Hero Callout
st.markdown("""
<div class="hero-banner">
    <h1>📡 Customer Lifetime Value & Churn Intelligence System</h1>
    <p>Predicting churn, estimating survival, computing CLV, and optimizing retention campaigns — all in one place.</p>
</div>
""", unsafe_allow_html=True)

# Verify whether precomputed pipeline outputs exist
df = load_master_data()
model, explainer = load_model_artifacts()
meta = load_meta()

if df is None:
    st.error("""
    **Pipeline outputs not found!**
    
    Please run the training pipeline first:
    ```bash
    python train_pipeline.py
    ```
    Then restart this app.
    """)
    st.stop()

# -----------------------------------------------------------------------------
# High-Level Portfolio KPI Cards
# -----------------------------------------------------------------------------
st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

total_customers = len(df)
churn_rate = df["churn_probability"].mean()
high_risk_count = (df["churn_probability"] >= 0.7).sum()
avg_clv = df["clv_estimate"].mean()
avg_survival = df["survival_months"].mean()
metrics = meta.get("metrics", {})

# Render 6 distinct operational metrics across the top row
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Avg Churn Probability", f"{churn_rate:.1%}")
col3.metric("High Risk Customers", f"{high_risk_count:,}", delta=f"{high_risk_count/total_customers:.1%} of total", delta_color="inverse")
col4.metric("Avg CLV (Rs.)", f"{avg_clv:,.0f}")
col5.metric("Avg Survival (months)", f"{avg_survival:.1f}")
col6.metric("Model ROC-AUC", f"{metrics.get('roc_auc', 'N/A')}")

st.markdown("---")

# -----------------------------------------------------------------------------
# Strategic Business Questions Section
# -----------------------------------------------------------------------------
st.markdown('<div class="section-header">The Four Business Questions We Answer</div>', unsafe_allow_html=True)

q_col1, q_col2, q_col3, q_col4 = st.columns(4)

with q_col1:
    st.markdown("""
    <div class="info-card">
        <h4>🚨 Who will leave?</h4>
        <p>LightGBM churn classifier predicts which customers are likely to cancel their subscription, with SHAP explanations.</p>
    </div>
    """, unsafe_allow_html=True)

with q_col2:
    st.markdown("""
    <div class="info-card">
        <h4>⏳ How long will they stay?</h4>
        <p>Kaplan-Meier & Cox PH survival models estimate the expected remaining tenure for each customer.</p>
    </div>
    """, unsafe_allow_html=True)

with q_col3:
    st.markdown("""
    <div class="info-card">
        <h4>💰 How much are they worth?</h4>
        <p>CLV is computed using survival-model predicted months and discounted cash flow analysis.</p>
    </div>
    """, unsafe_allow_html=True)

with q_col4:
    st.markdown("""
    <div class="info-card">
        <h4>🎯 How to retain them?</h4>
        <p>Uplift modeling identifies "persuadable" customers who will respond to offers — avoiding wasted discounts.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# Customer Segment Snapshot
# -----------------------------------------------------------------------------
st.markdown('<div class="section-header">Customer Segment Snapshot</div>', unsafe_allow_html=True)

# Compute segment aggregates dynamically
seg_summary = df.groupby("customer_segment").agg(
    Count=("customerID", "count"),
    Avg_Churn_Prob=("churn_probability", "mean"),
    Avg_CLV=("clv_estimate", "mean"),
    Avg_Tenure=("tenure", "mean"),
).round(2).reset_index()
seg_summary.columns = ["Segment", "Customers", "Avg Churn Prob", "Avg CLV (Rs.)", "Avg Tenure (mo)"]
seg_summary["Avg Churn Prob"] = seg_summary["Avg Churn Prob"].map("{:.1%}".format)
seg_summary["Avg CLV (Rs.)"] = seg_summary["Avg CLV (Rs.)"].map("{:,.0f}".format)
seg_summary["Avg Tenure (mo)"] = seg_summary["Avg Tenure (mo)"].map("{:.1f}".format)

st.dataframe(seg_summary, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Model Performance & Diagnostic Metrics
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)

m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
m_col1.metric("Accuracy",  f"{metrics.get('accuracy', 0):.2%}")
m_col2.metric("Precision", f"{metrics.get('precision', 0):.2%}")
m_col3.metric("Recall",    f"{metrics.get('recall', 0):.2%}")
m_col4.metric("F1 Score",  f"{metrics.get('f1', 0):.2%}")
m_col5.metric("ROC-AUC",   f"{metrics.get('roc_auc', 0):.4f}")

# -----------------------------------------------------------------------------
# Interactive Probability Distribution (Altair)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="section-header">Churn Probability Distribution</div>', unsafe_allow_html=True)

import altair as alt

chart_df = pd.DataFrame({"churn_probability": df["churn_probability"]})
hist = alt.Chart(chart_df).mark_bar(color="#667eea", opacity=0.8).encode(
    x=alt.X("churn_probability:Q", bin=alt.Bin(maxbins=40),
            title="Predicted Churn Probability"),
    y=alt.Y("count()", title="Customer Count"),
    tooltip=["count()"]
).properties(height=280).configure_axis(
    labelColor="#a0aec0", titleColor="#a0aec0"
).configure_view(stroke=None)

st.altair_chart(hist, use_container_width=True)

st.info("""
**How to use this app:**
- Navigate using the sidebar links above
- Start with **Exploratory Analysis** to understand the data
- Use **Churn Predictor** to look up individual customers
- Use **ROI Simulator** to model our retention campaign budget
""")
