"""
Page 3: Survival Analysis
==========================

Shows Kaplan-Meier survival curves and Cox PH model results.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.utils.feature_engineering import load_raw_data, prepare_survival_data

st.set_page_config(page_title="Survival Analysis | Churn Intelligence", page_icon="⏳", layout="wide")

st.markdown("""
<style>
    .section-header { font-size: 1.1rem; font-weight: 700; color: #667eea;
        border-bottom: 2px solid #667eea; padding-bottom: 6px; margin: 20px 0 14px; }
    [data-testid="metric-container"] { background: linear-gradient(135deg, #1e2130, #2a2f45);
        border-radius: 12px; padding: 12px 16px; border: 1px solid #3d4460; }
</style>
""", unsafe_allow_html=True)

st.title("⏳ Customer Survival Analysis")
st.caption("How long will customers stay? Kaplan-Meier + Cox Proportional Hazards models.")

st.info("""
**What is survival analysis in this context?**

In survival analysis, "survival" means a customer *remaining* with the company.
- **Duration**: months the customer has been with us (tenure)
- **Event**: churned = 1 (event happened), still active = 0 (censored)
- **Censoring** is key: customers who haven't churned *yet* tell us they survived
  at least this long — we can't ignore them!
""")

# Load
@st.cache_data
def get_survival_data():
    path = os.path.join(ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = load_raw_data(path)
    return prepare_survival_data(df)

@st.cache_data
def load_master():
    p = os.path.join(ROOT, "outputs", "master_predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

survival_df = get_survival_data()
df_master = load_master()

if survival_df is None:
    st.error("Run the pipeline first.")
    st.stop()

# ─── KM Overall ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Overall Customer Survival Curve</div>', unsafe_allow_html=True)

km_overall_img = os.path.join(ROOT, "outputs", "km_overall.png")
if os.path.exists(km_overall_img):
    st.image(km_overall_img, use_container_width=True)
else:
    # Re-generate on the fly
    kmf = KaplanMeierFitter()
    kmf.fit(survival_df["duration"], event_observed=survival_df["event"], label="All Customers")
    fig, ax = plt.subplots(figsize=(10, 5))
    kmf.plot_survival_function(ax=ax, ci_show=True, color="steelblue")
    ax.set_title("Kaplan-Meier Survival Curve")
    ax.set_xlabel("Months since joining")
    ax.set_ylabel("Survival Probability")
    ax.set_facecolor("#1e2130")
    fig.patch.set_facecolor("#1e2130")
    st.pyplot(fig)
    plt.close()

# Survival at key timepoints
kmf2 = KaplanMeierFitter()
kmf2.fit(survival_df["duration"], event_observed=survival_df["event"])
col1, col2, col3, col4 = st.columns(4)
for col, months in zip([col1, col2, col3, col4], [6, 12, 24, 48]):
    try:
        surv_prob = kmf2.survival_function_at_times([months]).values[0]
        col.metric(f"Survival at {months}m", f"{surv_prob:.1%}")
    except:
        col.metric(f"Survival at {months}m", "N/A")

# ─── KM by Contract ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Survival by Contract Type</div>', unsafe_allow_html=True)

km_contract_img = os.path.join(ROOT, "outputs", "km_by_contract.png")
if os.path.exists(km_contract_img):
    st.image(km_contract_img, use_container_width=True)
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Month-to-month": "crimson", "One year": "orange", "Two year": "steelblue"}
    for ctype, color in colors.items():
        mask = survival_df["Contract"] == ctype
        if mask.sum() == 0:
            continue
        kmf_t = KaplanMeierFitter()
        kmf_t.fit(survival_df.loc[mask, "duration"],
                  event_observed=survival_df.loc[mask, "event"],
                  label=f"{ctype} (n={mask.sum()})")
        kmf_t.plot_survival_function(ax=ax, ci_show=True, color=color)
    ax.set_title("Survival by Contract Type")
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival Probability")
    st.pyplot(fig)
    plt.close()

# ─── Interactive KM by user choice ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Interactive: Compare Survival by Group</div>', unsafe_allow_html=True)

group_col = st.selectbox(
    "Group customers by:",
    ["Contract", "gender", "has_internet", "is_senior", "has_family", "is_electronic_check"]
)

# Map numeric flags to labels for readability
display_df = survival_df.copy()
if group_col in ["has_internet", "is_senior", "has_family", "is_electronic_check"]:
    label_map = {1: "Yes", 0: "No"}
    display_df[group_col] = display_df[group_col].map(label_map).fillna(display_df[group_col])

groups = display_df[group_col].dropna().unique()
palette = plt.cm.Set2(np.linspace(0, 1, len(groups)))

fig, ax = plt.subplots(figsize=(10, 5))
for grp, color in zip(groups, palette):
    mask = display_df[group_col] == grp
    if mask.sum() < 10:
        continue
    kmf_grp = KaplanMeierFitter()
    kmf_grp.fit(
        display_df.loc[mask, "duration"],
        event_observed=display_df.loc[mask, "event"],
        label=f"{grp} (n={mask.sum()})"
    )
    kmf_grp.plot_survival_function(ax=ax, ci_show=False, color=color)

ax.set_title(f"Survival Curves by {group_col}", fontsize=12)
ax.set_xlabel("Months since joining")
ax.set_ylabel("Survival Probability")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.set_facecolor("#1e2130")
fig.patch.set_facecolor("#1e2130")
ax.tick_params(colors="white")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─── Cox PH Model ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Cox Proportional Hazards Model</div>', unsafe_allow_html=True)

cox_img = os.path.join(ROOT, "outputs", "cox_coefficients.png")
if os.path.exists(cox_img):
    st.image(cox_img, use_container_width=True)

st.markdown("""
**Interpreting the Cox model hazard ratios:**

| Feature | Hazard Ratio | Interpretation |
|---------|-------------|----------------|
| `contract_risk` | > 1 | Month-to-month contracts dramatically increase churn hazard |
| `monthly_charges` | > 1 | Higher charges slightly increase churn risk |
| `service_count` | < 1 | More services = stickier customer (lower hazard) |
| `has_family` | < 1 | Customers with partner/dependents churn less |
| `is_electronic_check` | > 1 | Electronic check payment is a risk indicator |

> Hazard Ratio > 1 = increases churn risk | < 1 = decreases churn risk
""")

# Re-fit Cox for the summary table
@st.cache_data
def get_cox_summary():
    cox_cols = ["duration","event","monthly_charges","contract_risk",
                "service_count","is_senior","has_family","is_electronic_check","has_internet"]
    cox_cols = [c for c in cox_cols if c in survival_df.columns]
    cox_df = survival_df[cox_cols].dropna()
    cox_df = cox_df[cox_df["duration"] > 0]
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="duration", event_col="event", show_progress=False)
    summary = cph.summary[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].round(4)
    summary.columns = ["Hazard Ratio","HR Lower 95%","HR Upper 95%","p-value"]
    return summary

cox_summary = get_cox_summary()
st.dataframe(cox_summary.style.background_gradient(subset=["Hazard Ratio"], cmap="RdYlGn_r"),
             use_container_width=True)

# ─── Predicted Survival Distribution ──────────────────────────────────────────
if df_master is not None and "survival_months" in df_master.columns:
    st.markdown("---")
    st.markdown('<div class="section-header">Predicted Survival Months Distribution</div>',
                unsafe_allow_html=True)

    import altair as alt
    surv_hist = alt.Chart(df_master).mark_bar(color="#667eea", opacity=0.8).encode(
        x=alt.X("survival_months:Q", bin=alt.Bin(maxbins=30), title="Predicted Survival (months)"),
        y=alt.Y("count()", title="Customers"),
        tooltip=["count()"]
    ).properties(height=250, title="Distribution of Predicted Customer Survival")
    st.altair_chart(surv_hist, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Median Survival", f"{df_master['survival_months'].median():.1f} months")
    col2.metric("Mean Survival", f"{df_master['survival_months'].mean():.1f} months")
    col3.metric("Min Survival", f"{df_master['survival_months'].min():.1f} months")

