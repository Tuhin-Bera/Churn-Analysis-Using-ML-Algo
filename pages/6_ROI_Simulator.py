"""
Page 6: ROI Simulator — Retention Campaign Optimizer
=====================================================

This module provides an interactive decision-support simulator for commercial marketing managers.
It bridges data science outputs (churn probability, survival tenure, and uplift scores) with
financial budgeting (marketing cost, gross contribution margin, and Return on Investment).

The Business Problem Solved:
Marketing teams frequently ask:
"If we offer a discount of Rs.200 per month for 3 months to 500 customers, will the campaign be profitable?"

Standard naive calculators assume 100% of targeted customers stay, leading to disastrous budget overruns.
Our simulator incorporates:
1. Target Customer Selection: Restrict targeting by churn risk, persona segment, or uplift category.
2. Causal Response Rate: Uses CausalML uplift scores (ITE) to estimate the actual percentage
   of customers who will stay *because* of the discount.
3. Margin Rate & Horizon: Accounts for telecom operating contribution margins (~30%) across predicted
   survival months rather than assuming gross revenue is profit.
4. Sensitivity & Budget Optimization:
   Dynamically generates sensitivity curves plotting discount size against projected percentage ROI,
   identifying the optimal marketing budget sweet spot.
5. Enterprise Audit Logging:
   Persists simulated campaign parameters directly to the relational database (`campaign_log`).
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

# Attempt to import database logging utility
try:
    from src.database.db_setup import get_engine, log_campaign
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

st.set_page_config(page_title="ROI Simulator | Churn Intelligence", page_icon="📈", layout="wide")

# Custom financial signal callout styling
st.markdown("""
<style>
    .section-header { font-size: 1.1rem; font-weight: 700; color: #667eea;
        border-bottom: 2px solid #667eea; padding-bottom: 6px; margin: 20px 0 14px; }
    [data-testid="metric-container"] { background: linear-gradient(135deg, #1e2130, #2a2f45);
        border-radius: 12px; padding: 12px 16px; border: 1px solid #3d4460; }
    .roi-positive { background: #c6f6d5; border-radius: 10px; padding: 16px;
                    color: #276749; font-weight: 700; font-size: 1.1rem; }
    .roi-negative { background: #fed7d7; border-radius: 10px; padding: 16px;
                    color: #c53030; font-weight: 700; font-size: 1.1rem; }
    .roi-neutral  { background: #bee3f8; border-radius: 10px; padding: 16px;
                    color: #2c5282; font-weight: 700; font-size: 1.1rem; }
    .formula-box { background: #1a1f35; border-radius: 8px; padding: 14px;
                   border: 1px solid #3d4460; margin: 10px 0; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.title("📈 ROI Simulator — Retention Campaign Optimizer")
st.caption("Design our retention campaign and see the expected cost vs. savings.")


# -----------------------------------------------------------------------------
# Data Ingestion & Caching
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():
    """Reads scored customer observations with churn probabilities and uplift metrics."""
    p = os.path.join(ROOT, "outputs", "master_predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

df = load_data()
if df is None:
    st.error("Model predictions missing. Please run `python train_pipeline.py` first.")
    st.stop()


# -----------------------------------------------------------------------------
# Financial Logic & Methodology Documentation
# -----------------------------------------------------------------------------

with st.expander("Financial Formulation: How Campaign ROI is Modeled (Click to Expand)"):
    st.markdown("""
    **Step 1: Define the Target Population**
    - Filter customer base by minimum churn probability threshold.
    - Optionally restrict to "Persuadable" accounts to eliminate wasted spend on Lost Causes and Sure Things.

    **Step 2: Estimate True Incremental Retention (CausalML)**
    - Standard calculators assume all targeted users stay. We use average uplift scores ($ITE$):
    $$\\text{Estimated Response Rate} = \\text{mean}(\\text{uplift\\_score}) \\times \\text{Confidence Factor}$$

    **Step 3: Total Marketing Campaign Cost**
    $$\\text{Total Cost} = \\text{Targeted Accounts} \\times (\\text{Monthly Discount} \\times \\text{Duration Months})$$

    **Step 4: Expected Incremental Margin (Benefit)**
    $$\\text{Retained Accounts} = \\text{Targeted Accounts} \\times \\text{Estimated Response Rate}$$
    $$\\text{Total Benefit} = \\text{Retained Accounts} \\times (\\text{Avg Monthly Spend} \\times \\text{Margin Rate}) \\times \\text{Avg Remaining Months}$$

    **Step 5: Net Return on Investment (ROI)**
    $$\\text{ROI (\\%)} = \\frac{\\text{Total Benefit} - \\text{Total Cost}}{\\text{Total Cost}} \\times 100$$
    """)


# -----------------------------------------------------------------------------
# Interactive Campaign Parameter Inputs
# -----------------------------------------------------------------------------

st.markdown('<div class="section-header">Campaign Configuration Parameters</div>', unsafe_allow_html=True)

cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

# Column 1: Targeting Criteria
with cfg_col1:
    st.markdown("**1. Targeting Filters**")
    churn_threshold = st.slider(
        "Minimum Churn Probability Threshold (%)",
        min_value=20, max_value=90, value=50, step=5,
        help="Only target accounts whose predicted churn probability equals or exceeds this cutoff."
    ) / 100

    persuadable_only = st.checkbox(
        "Target Persuadables only (Recommended)",
        value=True,
        help="Restricts offers to accounts where uplift modeling confirmed positive incremental response."
    )

    segments_to_target = st.multiselect(
        "Filter by customer segment:",
        df["customer_segment"].unique().tolist(),
        default=df["customer_segment"].unique().tolist()
    )

# Column 2: Offer Mechanics
with cfg_col2:
    st.markdown("**2. Promotional Offer Parameters**")
    discount_amount = st.slider(
        "Monthly Discount Voucher (Rs. per Account)",
        min_value=50, max_value=1000, value=200, step=50,
        help="Monetary value deducted from each recipient's monthly invoice."
    )

    discount_duration_months = st.slider(
        "Promotional Duration (Months)",
        min_value=1, max_value=12, value=3,
        help="Number of consecutive billing cycles the discount remains active."
    )

    total_discount_per_customer = discount_amount * discount_duration_months
    st.markdown(f"*Total discount commitment per account: Rs.{total_discount_per_customer:,}*")

# Column 3: Financial & Risk Assumptions
with cfg_col3:
    st.markdown("**3. Financial & Risk Parameters**")
    margin_rate = st.slider(
        "Contribution Margin Rate (%)",
        min_value=10, max_value=60, value=30, step=5,
        help="Operating profit margin after network and variable support costs."
    ) / 100

    max_budget = st.number_input(
        "Total Campaign Budget Cap (Rs.)",
        min_value=1000, max_value=5000000, value=100000, step=5000,
        help="Hard cap on total promotional marketing capital."
    )

    confidence_adj = st.slider(
        "Response Haircut / Conservatism Factor (%)",
        min_value=50, max_value=100, value=80, step=5,
        help="Applies a conservative downward adjustment to raw uplift predictions to account for operational friction."
    ) / 100


# -----------------------------------------------------------------------------
# Campaign Simulation Computation
# -----------------------------------------------------------------------------

# Step A: Filter matching accounts
target_df = df[df["churn_probability"] >= churn_threshold].copy()
target_df = target_df[target_df["customer_segment"].isin(segments_to_target)]
if persuadable_only:
    target_df = target_df[target_df["uplift_category"] == "Persuadable"]

# Step B: Enforce budget cap
cost_per_customer = total_discount_per_customer
max_customers_by_budget = int(max_budget / cost_per_customer) if cost_per_customer > 0 else len(target_df)
target_df = target_df.head(max_customers_by_budget)

n_targeted = len(target_df)
total_cost = n_targeted * cost_per_customer

# Step C: Estimate incremental uplift and net benefits
if n_targeted > 0:
    avg_uplift = target_df["uplift_score"].clip(lower=0).mean()
    adjusted_response_rate = avg_uplift * confidence_adj
    adjusted_response_rate = max(adjusted_response_rate, 0.05) if persuadable_only else adjusted_response_rate
    adjusted_response_rate = min(adjusted_response_rate, 0.80)

    expected_retained = n_targeted * adjusted_response_rate
    avg_clv = target_df["clv_estimate"].mean()
    avg_monthly = target_df["MonthlyCharges"].mean()
    avg_remaining_months = target_df["survival_months"].mean()

    # Total financial return over projected survival tenure
    monthly_margin = avg_monthly * margin_rate
    benefit = expected_retained * monthly_margin * avg_remaining_months

    roi_pct = (benefit - total_cost) / total_cost * 100 if total_cost > 0 else 0
    net_benefit = benefit - total_cost
else:
    avg_uplift = adjusted_response_rate = expected_retained = 0
    avg_clv = avg_monthly = avg_remaining_months = monthly_margin = benefit = 0
    roi_pct = net_benefit = 0
    total_cost = 0


# -----------------------------------------------------------------------------
# Simulation Results Dashboard
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Campaign Financial Forecast</div>', unsafe_allow_html=True)

if n_targeted == 0:
    st.warning("No customers match our targeting criteria. Try relaxing the filters.")
    st.stop()

# Operational volume KPIs
r_col1, r_col2, r_col3, r_col4 = st.columns(4)
r_col1.metric("Targeted Accounts", f"{n_targeted:,}")
r_col2.metric("Estimated Response Rate", f"{adjusted_response_rate:.1%}")
r_col3.metric("Expected Retained Accounts", f"{expected_retained:,.0f}")
r_col4.metric("Total Promotional Cost", f"Rs.{total_cost:,.0f}")

st.markdown("---")

# Financial return KPIs
roi_col1, roi_col2, roi_col3 = st.columns(3)
roi_col1.metric("Gross Preserved Margin", f"Rs.{benefit:,.0f}")
roi_col2.metric("Net Financial Gain", f"Rs.{net_benefit:,.0f}",
                delta=f"Rs.{net_benefit:,.0f}",
                delta_color="normal" if net_benefit > 0 else "inverse")
roi_col3.metric("Projected Campaign ROI", f"{roi_pct:.1f}%",
                delta="Profitable" if roi_pct > 0 else "Unprofitable",
                delta_color="normal" if roi_pct > 0 else "inverse")

# Executive recommendation banner
if roi_pct > 50:
    st.markdown(f"""
    <div class="roi-positive">
        CAMPAIGN STRONGLY APPROVED — Projected ROI of {roi_pct:.1f}%!
        Expected net profit: Rs.{net_benefit:,.0f} generated from a Rs.{total_cost:,.0f} promotional investment.
    </div>
    """, unsafe_allow_html=True)
elif roi_pct > 0:
    st.markdown(f"""
    <div class="roi-neutral">
        CAMPAIGN CONDITIONALLY VIABLE — Positive ROI of {roi_pct:.1f}%.
        Expected net profit: Rs.{net_benefit:,.0f}. Ensure customer service overhead does not exceed margin gains.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="roi-negative">
        CAMPAIGN NOT RECOMMENDED — Negative ROI of {roi_pct:.1f}%.
        The cost of the proposed discount exceeds the economic value of preserved customer lifetimes.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sensitivity Analysis: Discount Amount vs. Projected ROI
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Sensitivity Analysis: Discount Amount vs. ROI</div>', unsafe_allow_html=True)

discount_range = list(range(50, 1050, 50))
roi_values = []
net_values = []
for d in discount_range:
    d_total = d * discount_duration_months
    n_cap = min(n_targeted, int(max_budget / d_total)) if d_total > 0 else n_targeted
    cost = n_cap * d_total
    retained = n_cap * adjusted_response_rate
    ben = retained * monthly_margin * avg_remaining_months
    roi = (ben - cost) / cost * 100 if cost > 0 else 0
    roi_values.append(roi)
    net_values.append(ben - cost)

sens_df = pd.DataFrame({
    "Discount (Rs.)": discount_range,
    "ROI (%)": roi_values,
    "Net Benefit (Rs.)": net_values
})

roi_line = alt.Chart(sens_df).mark_line(color="#667eea", strokeWidth=2.5).encode(
    x=alt.X("Discount (Rs.):Q", title="Monthly Discount (Rs. per Account)"),
    y=alt.Y("ROI (%):Q", title="Projected Campaign ROI (%)"),
    tooltip=[
        alt.Tooltip("Discount (Rs.):Q"),
        alt.Tooltip("ROI (%):Q", format=".1f"),
        alt.Tooltip("Net Benefit (Rs.):Q", format=",.0f"),
    ]
)
zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
    color="red", strokeDash=[4, 4]
).encode(y="y:Q")

current_line = alt.Chart(
    pd.DataFrame({"x": [discount_amount]})
).mark_rule(color="yellow", strokeDash=[4, 4]).encode(x="x:Q")

st.altair_chart(roi_line + zero_line + current_line, use_container_width=True)
st.caption(f"Yellow dashed line = current discount (Rs.{discount_amount}). Red dashed line = break-even (0% ROI).")


# -----------------------------------------------------------------------------
# Budget Allocation Optimizer
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Budget Optimization: Maximizing Preserved Margin</div>',
            unsafe_allow_html=True)

budget_range = list(range(10000, 510000, 10000))
opt_data = []
for budget in budget_range:
    n_cap = min(n_targeted, int(budget / cost_per_customer)) if cost_per_customer > 0 else n_targeted
    cost = n_cap * cost_per_customer
    if cost == 0:
        continue
    retained = n_cap * adjusted_response_rate
    ben = retained * monthly_margin * avg_remaining_months
    opt_data.append({
        "Budget (Rs.)": budget,
        "Expected Retained": retained,
        "Net Benefit (Rs.)": ben - cost,
        "ROI (%)": (ben - cost) / cost * 100
    })

opt_df = pd.DataFrame(opt_data)
if len(opt_df) > 0:
    col1, col2 = st.columns(2)
    with col1:
        net_chart = alt.Chart(opt_df).mark_area(
            color="#667eea", opacity=0.5, line={"color": "#667eea"}
        ).encode(
            x=alt.X("Budget (Rs.):Q", title="Allocated Marketing Budget (Rs.)"),
            y=alt.Y("Net Benefit (Rs.):Q", title="Net Preserved Margin (Rs.)"),
            tooltip=["Budget (Rs.)", "Net Benefit (Rs.)", "Expected Retained"]
        ).properties(height=280, title="Net Preserved Profit vs. Total Budget")
        st.altair_chart(net_chart, use_container_width=True)
    with col2:
        roi_opt_chart = alt.Chart(opt_df).mark_line(color="#f6e05e", strokeWidth=2.5).encode(
            x=alt.X("Budget (Rs.):Q", title="Allocated Marketing Budget (Rs.)"),
            y=alt.Y("ROI (%):Q", title="Percentage ROI (%)"),
            tooltip=["Budget (Rs.)", "ROI (%)"]
        ).properties(height=280, title="Percentage ROI vs. Total Budget")
        zero_r = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            color="red", strokeDash=[4,4]
        ).encode(y="y:Q")
        st.altair_chart(roi_opt_chart + zero_r, use_container_width=True)


# -----------------------------------------------------------------------------
# Targeted Customer Roster & Campaign Export
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Targeted Customer Roster & CRM Export</div>', unsafe_allow_html=True)

show_cols = ["customerID","churn_probability","uplift_score","clv_estimate",
             "survival_months","customer_segment","MonthlyCharges","Contract"]
show_cols = [c for c in show_cols if c in target_df.columns]

st.markdown(f"**Identified {n_targeted:,} accounts selected for campaign activation:**")
st.dataframe(
    target_df[show_cols].style.format({
        "churn_probability": "{:.1%}",
        "uplift_score": "{:.3f}",
        "clv_estimate": "Rs.{:,.0f}",
        "survival_months": "{:.0f}",
        "MonthlyCharges": "Rs.{:.2f}",
    }),
    use_container_width=True, hide_index=True
)

csv = target_df[show_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Campaign Target Account Roster (CSV)",
    csv, "roi_campaign_targets.csv", "text/csv"
)


# -----------------------------------------------------------------------------
# Relational Database Campaign Audit Trail
# -----------------------------------------------------------------------------

if st.button("Persist Campaign Simulation to Relational Database", type="primary"):
    if DB_AVAILABLE:
        try:
            engine = get_engine()
            log_campaign(engine, total_discount_per_customer, n_targeted,
                         int(expected_retained), benefit)
            st.success(f"Simulation successfully logged to 'campaign_log' table! Projected ROI: {roi_pct:.1f}%")
        except Exception as e:
            st.error(f"Database logging failed: {e}")
    else:
        st.info("Relational database connection unavailable. Results rendered above.")
