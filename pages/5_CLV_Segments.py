"""
Page 5: CLV & Customer Segmentation
======================================

This module bridges financial valuation and behavioral grouping by combining
Customer Lifetime Value (CLV) with unsupervised K-Means clustering ($k=4$).

Analytical Architecture:
1. Discounted Cash Flow CLV:
   Calculates the present economic value of future operating margins for all 7,043 customers,
   incorporating predicted survival durations from the Lifelines Cox PH model.
   
2. Multi-Dimensional Customer Persona Segmentation:
   Partitions accounts into 4 clusters based on tenure, monthly fee, churn risk,
   service count, and lifetime value:
   - High-Value Loyal (VIP revenue drivers)
   - At-Risk Churners (Urgent win-back targets)
   - Low-Value Passive (Organic upsell candidates)
   - New Promising (Critical onboarding stage)
   
3. Pareto Value Concentration Analysis:
   Quantifies the proportion of cumulative portfolio CLV driven by the top decile of customers.
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

st.title("💰 Customer Lifetime Value & Behavioral Segmentation")
st.caption("Quantifying account equity via discounted cash flow modeling and profiling customer personas using K-Means clustering.")


# -----------------------------------------------------------------------------
# Data Loading & Caching
# -----------------------------------------------------------------------------

@st.cache_data
def load_master():
    """Reads scored customer observations with CLV estimates and cluster labels."""
    p = os.path.join(ROOT, "outputs", "master_predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

df = load_master()
if df is None:
    st.error("Predictions missing. Please run `python train_pipeline.py` first.")
    st.stop()


# -----------------------------------------------------------------------------
# Customer Lifetime Value Overview & Methodology
# -----------------------------------------------------------------------------

st.markdown('<div class="section-header">Customer Lifetime Value (CLV) Overview</div>', unsafe_allow_html=True)

with st.expander("Financial Methodology: How We Calculate CLV (Click to Expand)"):
    st.markdown(r"""
    **Finite Annuity Discounted Cash Flow Formula:**

    $$CLV = MonthlyCharges \times MarginRate \times \frac{1 - (1+r)^{-N}}{r}$$

    Where:
    - **MonthlyCharges:** Customer's monthly recurring revenue (MRR).
    - **MarginRate:** 30% contribution margin rate (standard benchmark for telecom fixed/mobile operations).
    - **r:** Monthly discount rate (10% annual cost of capital / 12 = 0.833% per month).
    - **N:** Expected remaining customer lifetime in months forecasted by the Cox Proportional Hazards model.
    - **Annuity Factor:** $\frac{1 - (1+r)^{-N}}{r}$ discounts future monthly cash flows to present value.
    """)

# Render top-level financial KPIs
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Portfolio Equity", f"Rs.{df['clv_estimate'].sum():,.0f}")
col2.metric("Mean Account CLV", f"Rs.{df['clv_estimate'].mean():,.0f}")
col3.metric("Median Account CLV", f"Rs.{df['clv_estimate'].median():,.0f}")
col4.metric("Maximum Account CLV", f"Rs.{df['clv_estimate'].max():,.0f}")
col5.metric("Standard Deviation", f"Rs.{df['clv_estimate'].std():,.0f}")

# Visual CLV distribution
clv_img = os.path.join(ROOT, "outputs", "clv_distribution.png")
if os.path.exists(clv_img):
    st.image(clv_img, use_container_width=True)
else:
    clv_hist = alt.Chart(df).mark_bar(color="#667eea", opacity=0.8).encode(
        x=alt.X("clv_estimate:Q", bin=alt.Bin(maxbins=40), title="Predicted CLV (Rs.)"),
        y=alt.Y("count()", title="Customer Count"),
        tooltip=["count()"]
    ).properties(height=280, title="Customer Lifetime Value Distribution")
    st.altair_chart(clv_hist, use_container_width=True)


# -----------------------------------------------------------------------------
# Risk vs. Value Bubble Chart
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Value-at-Risk Matrix (CLV vs. Churn Propensity)</div>', unsafe_allow_html=True)

scatter_df = df.sample(min(3000, len(df)), random_state=42)
color_scale = alt.Scale(
    domain=["High-Value Loyal","At-Risk Churners","Low-Value Passive","New Promising"],
    range=["#f6e05e","#fc8181","#81e6d9","#68d391"]
)
scatter = alt.Chart(scatter_df).mark_circle(size=35, opacity=0.6).encode(
    x=alt.X("churn_probability:Q", title="Predicted Churn Probability", scale=alt.Scale(domain=[0,1])),
    y=alt.Y("clv_estimate:Q", title="Predicted Customer Lifetime Value (Rs.)"),
    color=alt.Color("customer_segment:N", scale=color_scale,
                    legend=alt.Legend(title="Assigned Segment")),
    size=alt.Size("MonthlyCharges:Q", scale=alt.Scale(range=[20, 200]), title="Monthly Fee"),
    tooltip=["customerID","churn_probability","clv_estimate",
             "customer_segment","tenure","MonthlyCharges"]
).properties(height=380, title="Customer Value vs. Attrition Risk (Bubble Size = Monthly Spend)")
st.altair_chart(scatter, use_container_width=True)


# -----------------------------------------------------------------------------
# Behavioral Segmentation Diagnostics
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Behavioral Clusters (K-Means, k=4)</div>', unsafe_allow_html=True)

seg_img = os.path.join(ROOT, "outputs", "customer_segments.png")
if os.path.exists(seg_img):
    st.image(seg_img, use_container_width=True)


# -----------------------------------------------------------------------------
# Segment Persona Playbooks
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Segment Profiles & Operational Strategies</div>', unsafe_allow_html=True)

segment_info = {
    "High-Value Loyal": {
        "style": "seg-high",
        "description": "Long-tenure, high-spending customers with low churn probability. These are our best customers.",
        "strategy": "Reward with premium loyalty programs. Upsell to higher tiers. They're worth keeping at almost any cost.",
        "kpis": ["Highest CLV (Mean Rs.1,440)", "Lowest Churn Risk (~20%)", "Longest Tenure Cohort (>50 mo)"],
    },
    "At-Risk Churners": {
        "style": "seg-atrisk",
        "description": "Medium-to-high tenure customers who are showing churn signals — likely unhappy with price or service.",
        "strategy": "Proactive win-back campaigns. Targeted discounts. Personal outreach. Check SHAP explanations to understand drivers.",
        "kpis": ["Elevated Churn Risk (~59%)", "High Economic Equity (Mean Rs.1,052)", "Urgent Intervention Pool"],
    },
    "Low-Value Passive": {
        "style": "seg-low",
        "description": "Long-tenure, low-spending customers. They're loyal but don't generate much revenue.",
        "strategy": "Focus on upselling — bundle more services to increase ARPU. Low priority for expensive retention campaigns.",
        "kpis": ["Low Average Monthly Spend", "Minimal Churn Probability (~10%)", "Add-on Upsell Candidates"],
    },
    "New Promising": {
        "style": "seg-new",
        "description": "Short-tenure customers who are at moderate risk. Early churn prevention is critical here.",
        "strategy": "Strong onboarding experience. Early engagement. Prevent 'first 90 days' churn. Offer contract upgrades.",
        "kpis": ["Tenure < 12 Months", "Substantial Future Potential", "Onboarding Retention Target"],
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
            st.markdown(f"**Operational Playbook:** {seg_data['strategy']}")
        with scol2:
            st.markdown("**Key Operational Indicators:**")
            for kpi in seg_data["kpis"]:
                st.markdown(f"- {kpi}")
        with scol3:
            st.metric("Cohort Size", f"{len(seg_rows):,}")
            st.metric("Average CLV", f"Rs.{seg_rows['clv_estimate'].mean():,.0f}")
            st.metric("Cohort Churn Rate", f"{seg_rows['churn_probability'].mean():.1%}")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")


# -----------------------------------------------------------------------------
# Segment Customer Roster & CSV Export
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Segment Customer Explorer & Campaign Export</div>', unsafe_allow_html=True)

selected_seg = st.selectbox(
    "Filter and inspect customer roster by segment:",
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

csv = seg_df.to_csv(index=False).encode("utf-8")
st.download_button(
    f"Download {selected_seg} Customer List (CSV)",
    csv,
    f"segment_{selected_seg.replace(' ','_').lower()}.csv",
    "text/csv"
)


# -----------------------------------------------------------------------------
# Pareto Wealth Concentration Analysis
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Pareto Concentration & Percentile Analysis</div>', unsafe_allow_html=True)

percentiles = [10, 25, 50, 75, 90, 95, 99]
clv_percs = np.percentile(df["clv_estimate"], percentiles)
perc_df = pd.DataFrame({
    "Percentile Rank": [f"Top {100-p}% (P{p})" for p in percentiles],
    "Minimum CLV Threshold": [f"Rs.{v:,.0f}" for v in clv_percs]
})

col1, col2 = st.columns(2)
with col1:
    st.dataframe(perc_df, use_container_width=True, hide_index=True)
with col2:
    top10_clv = df.nlargest(int(len(df)*0.1), "clv_estimate")["clv_estimate"].sum()
    total_clv = df["clv_estimate"].sum()
    st.metric("Top 10% Accounts Equity Share", f"{top10_clv/total_clv:.1%} of Total Portfolio")
    st.metric("Top 10% Customer Count", f"{int(len(df)*0.1):,} Accounts")
    st.info("Classic 80-20 rule applies: focus our best retention efforts on the highest-CLV customers.")
