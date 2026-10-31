"""
Page 4: Uplift Modeling
=========================

This module implements Causal Inference and Uplift Modeling using CausalML.
The business question addressed here is:
"Which customers will stay BECAUSE of our offer, not despite it?"

Why Standard Churn Propensity Models Waste Marketing Capital:
Traditional churn models target accounts with the highest churn risk.
However, empirical marketing science demonstrates that:
1. Many high-risk accounts are "Lost Causes" who churn regardless of promotional incentives.
2. Many low-risk accounts are "Sure Things" who would stay organically anyway.
3. Blindly distributing discounts to all high-risk accounts unnecessarily depresses contribution margins.

The 4 Uplift Quadrants:
- Persuadables (High Risk, Positive Uplift): The ONLY cohort where an incentive produces positive incremental ROI.
- Lost Causes (High Risk, Non-Positive Uplift): Will leave regardless; do not waste promotional discount budget.
- Sure Things (Low Risk, Positive/Neutral Uplift): Organic loyalty; discounting them simply erodes margin.
- Do Not Disturb (Low Risk, Negative Uplift): Customers whose cancellation hazard increases if disturbed.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

st.set_page_config(page_title="Uplift Modeling | Churn Intelligence", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .section-header { font-size: 1.1rem; font-weight: 700; color: #667eea;
        border-bottom: 2px solid #667eea; padding-bottom: 6px; margin: 20px 0 14px; }
    [data-testid="metric-container"] { background: linear-gradient(135deg, #1e2130, #2a2f45);
        border-radius: 12px; padding: 12px 16px; border: 1px solid #3d4460; }
    .quadrant-card { border-radius: 10px; padding: 14px; margin: 5px 0; }
    .q-persuadable { background: #c6f6d5; border-left: 4px solid #38a169; }
    .q-lost { background: #fed7d7; border-left: 4px solid #e53e3e; }
    .q-sure { background: #bee3f8; border-left: 4px solid #3182ce; }
    .q-dnd { background: #e2e8f0; border-left: 4px solid #718096; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Uplift Modeling — Who Responds to Offers?")
st.caption("CausalML Meta-Learner: Identifying the responsive customer cohort to maximize retention marketing efficiency.")


# -----------------------------------------------------------------------------
# Uplift Theory & Mathematical Framing
# -----------------------------------------------------------------------------

with st.expander("📚 Theoretical Framework: The Four Uplift Quadrants (Click to Expand)"):
    st.markdown("""
    ### Why Standard Churn Targeting Fails
    If we just target the **highest-risk churners** with discounts, we waste money on:
    - **Lost Causes** — they'll churn regardless
    - **Do-Not-Disturb** — our offer might actually irritate them
    - **Sure Things** — they'd stay anyway

    ### The Four Behavioral Response Quadrants

    | Quadrant | Churn Risk | Response to Retention Offer | Prescribed Marketing Strategy |
    |----------|-----------|-----------------------------|-------------------------------|
    | **Persuadable** | High (>40%) | Positive incremental response | **PRIORITY CAMPAIGN TARGET** |
    | **Lost Cause** | High (>40%) | No behavioral change | Avoid discounts; resolve technical tickets |
    | **Sure Thing** | Low (<=40%) | Stays regardless | Maintain standard service pricing |
    | **Do Not Disturb** | Low (<=40%) | Negative reaction to contact | Avoid unsolicited promotional outreach |

    ### Methodological Note on Observational Data
    True causal inference requires randomized controlled trials (A/B testing).
    Because observational telecom datasets lack randomized treatment columns, we applied an
    S-Learner meta-regressor trained on simulated treatment assignments matching realistic enterprise CRM rules.
    """)


# -----------------------------------------------------------------------------
# Data Loading & Overview
# -----------------------------------------------------------------------------

@st.cache_data
def load_master():
    """Reads scored customer observations with CausalML uplift scores and categories."""
    p = os.path.join(ROOT, "outputs", "master_predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

df = load_master()
if df is None:
    st.error("Model outputs missing. Run `python train_pipeline.py` first.")
    st.stop()


# -----------------------------------------------------------------------------
# Quadrant Distribution KPI Cards
# -----------------------------------------------------------------------------

st.markdown('<div class="section-header">Customer Distribution Across Uplift Quadrants</div>', unsafe_allow_html=True)

cat_counts = df["uplift_category"].value_counts()

col1, col2, col3, col4 = st.columns(4)

with col1:
    n = cat_counts.get("Persuadable", 0)
    st.metric("Persuadable Cohort", f"{n:,}", help="High churn risk, will stay if offered retention incentive")
with col2:
    n = cat_counts.get("Lost Cause", 0)
    st.metric("Lost Cause Cohort", f"{n:,}", help="High churn risk, but unresponsive to promotional offers")
with col3:
    n = cat_counts.get("Sure Thing Cohort", 0)
    st.metric("Sure Things", f"{cat_counts.get('Sure Thing', 0):,}", help="Low churn risk, organic customer retention")
with col4:
    n = cat_counts.get("Do Not Disturb", 0)
    st.metric("Do Not Disturb", f"{n:,}", help="Low churn risk, best left untouched by aggressive marketing")

# Pre-rendered uplift distribution figure
uplift_img = os.path.join(ROOT, "outputs", "uplift_distribution.png")
if os.path.exists(uplift_img):
    st.image(uplift_img, use_container_width=True)


# -----------------------------------------------------------------------------
# 2D Decision Matrix: Churn Risk vs. Uplift Score
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">The 2x2 Decision Matrix: Churn Probability vs. Uplift Score</div>',
            unsafe_allow_html=True)

scatter_df = df.sample(min(2000, len(df)), random_state=42)
color_scale = alt.Scale(
    domain=["Persuadable", "Lost Cause", "Sure Thing", "Do Not Disturb"],
    range=["#38a169", "#e53e3e", "#3182ce", "#718096"]
)
scatter = alt.Chart(scatter_df).mark_circle(size=40, opacity=0.6).encode(
    x=alt.X("churn_probability:Q", title="Predicted Churn Probability", scale=alt.Scale(domain=[0, 1])),
    y=alt.Y("uplift_score:Q", title="Estimated Uplift Score (ITE)"),
    color=alt.Color("uplift_category:N", scale=color_scale, legend=alt.Legend(title="Assigned Quadrant")),
    tooltip=["customerID", "churn_probability", "uplift_score", "uplift_category",
             "customer_segment", "clv_estimate"]
).properties(height=400, title="Individual Treatment Effect (ITE) vs. Churn Propensity (2,000 Sampled Accounts)")

# Threshold lines separating quadrants
zero_line = alt.Chart(pd.DataFrame({"y": [0.05]})).mark_rule(
    color="yellow", strokeDash=[4, 4], opacity=0.7
).encode(y="y:Q")

cutoff_line = alt.Chart(pd.DataFrame({"x": [0.4]})).mark_rule(
    color="white", strokeDash=[4, 4], opacity=0.7
).encode(x="x:Q")

st.altair_chart(scatter + zero_line + cutoff_line, use_container_width=True)
st.caption("Upper-Right Quadrant (Green) represents true Persuadable accounts: high attrition risk paired with high offer sensitivity.")


# -----------------------------------------------------------------------------
# Priority Target List: The Persuadable Accounts
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Persuadable Customer Roster (Primary Campaign Targets)</div>',
            unsafe_allow_html=True)

persuadable_df = df[df["uplift_category"] == "Persuadable"].sort_values(
    "uplift_score", ascending=False
)[["customerID","churn_probability","uplift_score","clv_estimate",
   "survival_months","customer_segment","Contract","tenure","MonthlyCharges"]].head(100)

st.markdown(f"**Identified {len(df[df['uplift_category']=='Persuadable']):,} high-priority accounts.** These accounts combine elevated departure risk with verified offer responsiveness.")

st.dataframe(
    persuadable_df.style.format({
        "churn_probability": "{:.1%}",
        "uplift_score": "{:.3f}",
        "clv_estimate": "Rs.{:,.0f}",
        "survival_months": "{:.0f}",
        "MonthlyCharges": "Rs.{:.2f}",
    }).background_gradient(subset=["uplift_score"], cmap="Greens"),
    use_container_width=True, hide_index=True
)


# -----------------------------------------------------------------------------
# Portfolio Economic Value at Stake
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Economic Value at Stake in the Persuadable Cohort</div>',
            unsafe_allow_html=True)

persuadable_full = df[df["uplift_category"] == "Persuadable"]
total_clv_at_stake = persuadable_full["clv_estimate"].sum()
avg_clv_persuadable = persuadable_full["clv_estimate"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Persuadable Accounts", f"{len(persuadable_full):,}")
c2.metric("Total Cumulative CLV at Stake", f"Rs.{total_clv_at_stake:,.0f}")
c3.metric("Average CLV per Persuadable Account", f"Rs.{avg_clv_persuadable:,.0f}")

st.info(f"""
**Strategic Business Impact:**
If an orchestrated campaign successfully retains 50% of the {len(persuadable_full)} persuadable subscribers,
the company preserves approximately **Rs.{total_clv_at_stake * 0.5:,.0f}** in projected customer lifetime contribution.

To model the cost-benefit trade-off and test specific discount parameters, proceed to the **ROI Simulator** page.
""")
