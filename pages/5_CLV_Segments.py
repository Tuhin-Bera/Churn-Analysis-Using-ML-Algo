"""
Page 5: CLV & Customer Segmentation
======================================
Author: Tuhin Bera | Data Science Intern
Date: October 2026

Shows CLV distribution and K-Means customer segments with actionable insights.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

st.set_page_config(page_title="CLV & Segments | Churn Intelligence", page_icon="💰", layout="wide")

st.markdown("""
<style>
    .section-header { font-size: 1.1rem; font-weight: 700; color: #667eea;
        border-bottom: 2px solid #667eea; padding-bottom: 6px; margin: 20px 0 14px; }
    [data-testid="metric-container"] { background: linear-gradient(135deg, #1e2130, #2a2f45);
        border-radius: 12px; padding: 12px 16px; border: 1px solid #3d4460; }
    .seg-card { border-radius: 12px; padding: 18px; margin: 8px 0; }
    .seg-high    { background: #1a2a3a; border-left: 4px solid #f6e05e; }
    .seg-atrisk  { background: #2a1a1a; border-left: 4px solid #fc8181; }
    .seg-low     { background: #1a2a2a; border-left: 4px solid #81e6d9; }
    .seg-new     { background: #1a2a1a; border-left: 4px solid #68d391; }
</style>
""", unsafe_allow_html=True)

st.title("💰 Customer Lifetime Value & Segmentation")
st.caption("How much is each customer worth? And which group do they belong to?")

# ─── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_master():
    p = os.path.join(ROOT, "outputs", "master_predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

df = load_master()
if df is None:
    st.error("Run `python train_pipeline.py` first.")
    st.stop()

# ─── CLV Overview ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Customer Lifetime Value Overview</div>', unsafe_allow_html=True)

with st.expander("How is CLV computed?"):
    st.markdown(r"""
    **CLV Formula (Discounted Cash Flow approach):**

    $$CLV = MonthlyCharges \times MarginRate \times \frac{1 - (1+r)^{-N}}{r}$$

    Where:
    - **MarginRate** = 30% (telecom industry average)
    - **r** = monthly discount rate (10% annual / 12 = 0.833%/month)
    - **N** = predicted remaining months from the Cox PH survival model

    This gives the present value of expected future cash flows from each customer.
    """)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Portfolio CLV", f"Rs.{df['clv_estimate'].sum():,.0f}")
col2.metric("Avg CLV per Customer", f"Rs.{df['clv_estimate'].mean():,.0f}")
col3.metric("Median CLV", f"Rs.{df['clv_estimate'].median():,.0f}")
col4.metric("Max CLV", f"Rs.{df['clv_estimate'].max():,.0f}")
col5.metric("CLV Std Dev", f"Rs.{df['clv_estimate'].std():,.0f}")

# CLV distribution chart
clv_img = os.path.join(ROOT, "outputs", "clv_distribution.png")
if os.path.exists(clv_img):
    st.image(clv_img, use_container_width=True)
else:
    clv_hist = alt.Chart(df).mark_bar(color="#667eea", opacity=0.8).encode(
        x=alt.X("clv_estimate:Q", bin=alt.Bin(maxbins=40), title="CLV Estimate (Rs.)"),
        y=alt.Y("count()", title="Customers"),
        tooltip=["count()"]
    ).properties(height=280, title="CLV Distribution")
    st.altair_chart(clv_hist, use_container_width=True)

# ─── CLV vs Churn ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">CLV vs Churn Risk — Value at Risk</div>', unsafe_allow_html=True)

scatter_df = df.sample(min(3000, len(df)), random_state=42)
color_scale = alt.Scale(
    domain=["High-Value Loyal","At-Risk Churners","Low-Value Passive","New Promising"],
    range=["#f6e05e","#fc8181","#81e6d9","#68d391"]
)
scatter = alt.Chart(scatter_df).mark_circle(size=35, opacity=0.6).encode(
    x=alt.X("churn_probability:Q", title="Churn Probability", scale=alt.Scale(domain=[0,1])),
    y=alt.Y("clv_estimate:Q", title="CLV Estimate (Rs.)"),
    color=alt.Color("customer_segment:N", scale=color_scale,
                    legend=alt.Legend(title="Segment")),
    size=alt.Size("MonthlyCharges:Q", scale=alt.Scale(range=[20, 200])),
    tooltip=["customerID","churn_probability","clv_estimate",
             "customer_segment","tenure","MonthlyCharges"]
).properties(height=380, title="CLV vs Churn Risk (bubble size = monthly charges)")
st.altair_chart(scatter, use_container_width=True)

# ─── Customer Segments ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Customer Segments (K-Means, k=4)</div>', unsafe_allow_html=True)

seg_img = os.path.join(ROOT, "outputs", "customer_segments.png")
if os.path.exists(seg_img):
    st.image(seg_img, use_container_width=True)

# ─── Segment Profiles ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Segment Profiles & Strategies</div>', unsafe_allow_html=True)

segment_info = {
    "High-Value Loyal": {
        "style": "seg-high",
        "description": "Long-tenure, high-spending customers with low churn probability. These are your best customers.",
        "strategy": "Reward with premium loyalty programs. Upsell to higher tiers. They're worth keeping at almost any cost.",
        "kpis": ["Highest CLV", "Lowest churn risk", "Longest tenure"],
    },
    "At-Risk Churners": {
        "style": "seg-atrisk",
        "description": "Medium-to-high tenure customers who are showing churn signals — likely unhappy with price or service.",
        "strategy": "Proactive win-back campaigns. Targeted discounts. Personal outreach. Check SHAP explanations to understand drivers.",
        "kpis": ["High churn probability", "Medium-high CLV", "Key campaign targets"],
    },
    "Low-Value Passive": {
        "style": "seg-low",
        "description": "Long-tenure, low-spending customers. They're loyal but don't generate much revenue.",
        "strategy": "Focus on upselling — bundle more services to increase ARPU. Low priority for expensive retention campaigns.",
        "kpis": ["Lowest CLV", "Low churn risk", "Upsell opportunity"],
    },
    "New Promising": {
        "style": "seg-new",
        "description": "Short-tenure customers who are at moderate risk. Early churn prevention is critical here.",
        "strategy": "Strong onboarding experience. Early engagement. Prevent 'first 90 days' churn. Offer contract upgrades.",
        "kpis": ["Low tenure", "Moderate churn risk", "Long-term potential"],
    },
}

for seg_name, seg_data in segment_info.items():
    seg_rows = df[df["customer_segment"] == seg_name]
    if len(seg_rows) == 0:
        continue

    with st.container():
        st.markdown(f'<div class="seg-card {seg_data["style"]}">', unsafe_allow_html=True)
        scol1, scol2, scol3 = st.columns([2, 2, 1])
        with scol1:
            st.markdown(f"### {seg_name}")
            st.markdown(f"*{seg_data['description']}*")
            st.markdown(f"**Strategy:** {seg_data['strategy']}")
        with scol2:
            st.markdown("**KPIs:**")
            for kpi in seg_data["kpis"]:
                st.markdown(f"- {kpi}")
        with scol3:
            st.metric("Count", f"{len(seg_rows):,}")
            st.metric("Avg CLV", f"Rs.{seg_rows['clv_estimate'].mean():,.0f}")
            st.metric("Avg Churn Prob", f"{seg_rows['churn_probability'].mean():.1%}")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")

# ─── Segment Explorer ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Segment Explorer</div>', unsafe_allow_html=True)

selected_seg = st.selectbox(
    "View customers in segment:",
    df["customer_segment"].unique().tolist()
)

seg_df = df[df["customer_segment"] == selected_seg][
    ["customerID","tenure","MonthlyCharges","churn_probability","clv_estimate",
     "survival_months","uplift_category","Contract","Churn"]
].sort_values("clv_estimate", ascending=False)

st.dataframe(
    seg_df.head(100).style.format({
        "churn_probability": "{:.1%}",
        "clv_estimate": "Rs.{:,.0f}",
        "survival_months": "{:.0f}",
        "MonthlyCharges": "Rs.{:.2f}",
    }),
    use_container_width=True, hide_index=True
)

# Download button for segment data
csv = seg_df.to_csv(index=False).encode("utf-8")
st.download_button(
    f"Download {selected_seg} Customer List (CSV)",
    csv,
    f"segment_{selected_seg.replace(' ','_').lower()}.csv",
    "text/csv"
)

# ─── CLV Percentile Analysis ──────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">CLV Percentile Analysis</div>', unsafe_allow_html=True)

percentiles = [10, 25, 50, 75, 90, 95, 99]
clv_percs = np.percentile(df["clv_estimate"], percentiles)
perc_df = pd.DataFrame({"Percentile": [f"P{p}" for p in percentiles],
                         "CLV (Rs.)": [f"Rs.{v:,.0f}" for v in clv_percs]})

col1, col2 = st.columns(2)
with col1:
    st.dataframe(perc_df, use_container_width=True, hide_index=True)
with col2:
    top10_clv = df.nlargest(int(len(df)*0.1), "clv_estimate")["clv_estimate"].sum()
    total_clv = df["clv_estimate"].sum()
    st.metric("Top 10% customers contribute", f"{top10_clv/total_clv:.1%} of total CLV")
    st.metric("Top 10% customers count", f"{int(len(df)*0.1):,}")
    st.info("Classic 80-20 rule applies: focus your best retention efforts on the highest-CLV customers.")

