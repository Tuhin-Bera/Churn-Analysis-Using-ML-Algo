"""
Page 4: Uplift Modeling
=========================

Uplift modeling: who will respond to our offer?
Not all at-risk customers should get discounts.
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
st.caption("CausalML S-Learner: Identify customers most likely to stay BECAUSE of a retention offer.")

# ─── Theory explanation ────────────────────────────────────────────────────────
with st.expander("📚 Understanding Uplift Modeling (click to expand)"):
    st.markdown("""
    ### Why standard churn prediction isn't enough

    If we just target the **highest-risk churners** with discounts, we waste money on:
    - **Lost Causes** — they'll churn regardless
    - **Do-Not-Disturb** — our offer might actually irritate them
    - **Sure Things** — they'd stay anyway

    ### The four uplift quadrants

    | Quadrant | Churn Risk | Response to Offer | Action |
    |----------|-----------|-------------------|--------|
    | **Persuadable** | High | Would respond positively | PRIORITY TARGET |
    | **Lost Cause** | High | Won't respond | Don't waste budget |
    | **Sure Thing** | Low | Would respond, but not needed | Save money |
    | **Do Not Disturb** | Low | Offer might hurt | Avoid |

    ### How we estimated uplift (data limitation note)

    The IBM Telco dataset doesn't have randomized treatment/control data.
    So we used a **meta-learner (S-Learner)** approach:
    1. Simulated treatment assignment (which customers received an offer) using business logic
    2. Simulated post-offer outcomes with a realistic 20% lift effect
    3. Trained a CausalML S-Learner to estimate Individual Treatment Effects (ITE)
    
    In a real production setting, we would run an A/B test to get true causal data.
    """)

# ─── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_master():
    p = os.path.join(ROOT, "outputs", "master_predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

df = load_master()
if df is None:
    st.error("Run `python train_pipeline.py` first.")
    st.stop()

# ─── Uplift Overview ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Uplift Category Distribution</div>', unsafe_allow_html=True)

cat_counts = df["uplift_category"].value_counts()

col1, col2, col3, col4 = st.columns(4)
colors_map = {"Persuadable": "green", "Lost Cause": "red",
              "Sure Thing": "blue", "Do Not Disturb": "gray"}

with col1:
    n = cat_counts.get("Persuadable", 0)
    st.metric("Persuadable", f"{n:,}", help="High risk, will respond to offer")
with col2:
    n = cat_counts.get("Lost Cause", 0)
    st.metric("Lost Cause", f"{n:,}", help="High risk, won't respond")
with col3:
    n = cat_counts.get("Sure Thing", 0)
    st.metric("Sure Thing", f"{n:,}", help="Low risk, would respond but don't need to")
with col4:
    n = cat_counts.get("Do Not Disturb", 0)
    st.metric("Do Not Disturb", f"{n:,}", help="Low risk, don't engage with offer")

# ─── Uplift distribution images ────────────────────────────────────────────────
uplift_img = os.path.join(ROOT, "outputs", "uplift_distribution.png")
if os.path.exists(uplift_img):
    st.image(uplift_img, use_container_width=True)

# ─── Scatter: Churn Risk vs Uplift Score ──────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Churn Risk vs Uplift Score — The 2x2 Matrix</div>',
            unsafe_allow_html=True)

scatter_df = df.sample(min(2000, len(df)), random_state=42)
color_scale = alt.Scale(
    domain=["Persuadable", "Lost Cause", "Sure Thing", "Do Not Disturb"],
    range=["#38a169", "#e53e3e", "#3182ce", "#718096"]
)
scatter = alt.Chart(scatter_df).mark_circle(size=40, opacity=0.6).encode(
    x=alt.X("churn_probability:Q", title="Churn Probability", scale=alt.Scale(domain=[0, 1])),
    y=alt.Y("uplift_score:Q", title="Uplift Score (ITE)"),
    color=alt.Color("uplift_category:N", scale=color_scale, legend=alt.Legend(title="Category")),
    tooltip=["customerID", "churn_probability", "uplift_score", "uplift_category",
             "customer_segment", "clv_estimate"]
).properties(height=400, title="Churn Risk vs Uplift Score (Individual Treatment Effect)")

# Quadrant lines
zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
    color="white", strokeDash=[4, 4], opacity=0.5
).encode(y="y:Q")

cutoff_line = alt.Chart(pd.DataFrame({"x": [0.4]})).mark_rule(
    color="white", strokeDash=[4, 4], opacity=0.5
).encode(x="x:Q")

st.altair_chart(scatter + zero_line + cutoff_line, use_container_width=True)

# ─── Persuadable customer table ────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Persuadable Customers — Priority Retention Targets</div>',
            unsafe_allow_html=True)

persuadable_df = df[df["uplift_category"] == "Persuadable"].sort_values(
    "uplift_score", ascending=False
)[["customerID","churn_probability","uplift_score","clv_estimate",
   "survival_months","customer_segment","Contract","tenure","MonthlyCharges"]].head(100)

st.markdown(f"**{len(persuadable_df)} high-priority customers found.** These customers are at high churn risk AND likely to respond positively to a retention offer.")

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

# ─── CLV of Persuadables ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Revenue at Stake — CLV of Persuadable Customers</div>',
            unsafe_allow_html=True)

persuadable_full = df[df["uplift_category"] == "Persuadable"]
total_clv_at_stake = persuadable_full["clv_estimate"].sum()
avg_clv_persuadable = persuadable_full["clv_estimate"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Persuadable Customers", f"{len(persuadable_full):,}")
c2.metric("Total CLV at Stake", f"Rs.{total_clv_at_stake:,.0f}")
c3.metric("Average CLV per Persuadable", f"Rs.{avg_clv_persuadable:,.0f}")

st.info(f"""
**Business Impact:**
If we can retain even 50% of the {len(persuadable_full)} persuadable customers,
we could preserve Rs.{total_clv_at_stake * 0.5:,.0f} in lifetime value.

This justifies investing in targeted retention campaigns — go to the **ROI Simulator** page
to estimate the exact cost-benefit of different discount scenarios.
""")

