"""
Page 6: ROI Simulator
======================

The crown jewel of the project — the ROI calculator.
Business users can simulate: "If we offer a Rs.X discount to Y customers,
what is our expected return?"

The simulator uses uplift scores to estimate how many customers will actually
respond (rather than assuming all will), making it much more realistic than
a naive ROI calculation.
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

try:
    from src.database.db_setup import get_engine, log_campaign
    DB_AVAILABLE = True
except:
    DB_AVAILABLE = False

st.set_page_config(page_title="ROI Simulator | Churn Intelligence", page_icon="📈", layout="wide")

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

# ─── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    p = os.path.join(ROOT, "outputs", "master_predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

df = load_data()
if df is None:
    st.error("Run `python train_pipeline.py` first.")
    st.stop()

# ─── ROI Formula Explanation ──────────────────────────────────────────────────
with st.expander("How is ROI calculated? (click to see formula)"):
    st.markdown("""
    **Step 1: Identify target customers**
    - Filter by churn probability threshold
    - Optionally restrict to "Persuadable" customers (uplift > 0)
    
    **Step 2: Estimate response rate**
    - Use uplift scores to estimate the fraction who will respond positively
    - Response rate = avg(uplift_score) for targeted customers (clipped to [0, 1])
    
    **Step 3: Calculate campaign cost**
    ```
    Campaign Cost = Discount Amount × Number of Customers Targeted
    ```
    
    **Step 4: Estimate benefit**
    ```
    Expected Retained = Customers Targeted × Response Rate
    Benefit = Expected Retained × Avg CLV × Margin Rate
    ```
    
    **Step 5: Compute ROI**
    ```
    ROI = (Benefit - Cost) / Cost × 100%
    ```
    
    > **Note:** This is a simplified model. Real ROI calculations should also account
    > for the cost of customer service calls, campaign operational costs, and
    > probability that retained customers generate their full CLV.
    """)

# ─── Campaign Configuration ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Campaign Configuration</div>', unsafe_allow_html=True)

cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

with cfg_col1:
    st.markdown("**Targeting Criteria**")
    churn_threshold = st.slider(
        "Min Churn Probability to Target (%)",
        min_value=20, max_value=90, value=50, step=5
    ) / 100

    persuadable_only = st.checkbox(
        "Target Persuadable customers only (recommended)",
        value=True,
        help="Persuadable = high churn risk AND positive uplift score. Most efficient targeting."
    )

    segments_to_target = st.multiselect(
        "Restrict to segment(s):",
        df["customer_segment"].unique().tolist(),
        default=df["customer_segment"].unique().tolist()
    )

with cfg_col2:
    st.markdown("**Offer Parameters**")
    discount_amount = st.slider(
        "Discount Amount per Customer (Rs.)",
        min_value=50, max_value=1000, value=200, step=50
    )

    discount_duration_months = st.slider(
        "Discount Duration (months)",
        min_value=1, max_value=12, value=3
    )

    total_discount_per_customer = discount_amount * discount_duration_months
    st.markdown(f"*Total discount value per customer: Rs.{total_discount_per_customer:,}*")

with cfg_col3:
    st.markdown("**Business Assumptions**")
    margin_rate = st.slider(
        "Contribution Margin Rate (%)",
        min_value=10, max_value=60, value=30, step=5
    ) / 100

    max_budget = st.number_input(
        "Campaign Budget Cap (Rs.)",
        min_value=1000, max_value=5000000, value=100000, step=5000
    )

    confidence_adj = st.slider(
        "Confidence Adjustment (%)",
        min_value=50, max_value=100, value=80, step=5,
        help="Apply a haircut to the response rate for conservatism"
    ) / 100

# ─── Compute ROI ──────────────────────────────────────────────────────────────
# Filter target customers
target_df = df[df["churn_probability"] >= churn_threshold].copy()
target_df = target_df[target_df["customer_segment"].isin(segments_to_target)]
if persuadable_only:
    target_df = target_df[target_df["uplift_category"] == "Persuadable"]

# Apply budget cap
cost_per_customer = total_discount_per_customer
max_customers_by_budget = int(max_budget / cost_per_customer) if cost_per_customer > 0 else len(target_df)
target_df = target_df.head(max_customers_by_budget)

n_targeted = len(target_df)
total_cost = n_targeted * cost_per_customer

# Estimate response rate from uplift scores
if n_targeted > 0:
    avg_uplift = target_df["uplift_score"].clip(lower=0).mean()
    adjusted_response_rate = avg_uplift * confidence_adj
    # Floor at 5% if there are some persuadables (assumes minimum viable response)
    adjusted_response_rate = max(adjusted_response_rate, 0.05) if persuadable_only else adjusted_response_rate
    adjusted_response_rate = min(adjusted_response_rate, 0.80)  # cap at 80% (realistic)

    expected_retained = n_targeted * adjusted_response_rate
    avg_clv = target_df["clv_estimate"].mean()
    avg_monthly = target_df["MonthlyCharges"].mean()

    # Benefit = retained customers * avg remaining monthly margin * remaining months
    # Using CLV directly is more accurate than monthly * months
    avg_remaining_months = target_df["survival_months"].mean()
    monthly_margin = avg_monthly * margin_rate
    benefit = expected_retained * monthly_margin * avg_remaining_months

    roi_pct = (benefit - total_cost) / total_cost * 100 if total_cost > 0 else 0
    net_benefit = benefit - total_cost
else:
    avg_uplift = adjusted_response_rate = expected_retained = 0
    avg_clv = avg_monthly = avg_remaining_months = monthly_margin = benefit = 0
    roi_pct = net_benefit = 0
    total_cost = 0

# ─── Results Dashboard ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Campaign Results Forecast</div>', unsafe_allow_html=True)

if n_targeted == 0:
    st.warning("No customers match our targeting criteria. Try relaxing the filters.")
    st.stop()

r_col1, r_col2, r_col3, r_col4 = st.columns(4)
r_col1.metric("Customers Targeted", f"{n_targeted:,}")
r_col2.metric("Est. Response Rate", f"{adjusted_response_rate:.1%}")
r_col3.metric("Expected Retained", f"{expected_retained:,.0f}")
r_col4.metric("Total Campaign Cost", f"Rs.{total_cost:,.0f}")

st.markdown("---")

roi_col1, roi_col2, roi_col3 = st.columns(3)
roi_col1.metric("Expected Benefit", f"Rs.{benefit:,.0f}")
roi_col2.metric("Net Benefit", f"Rs.{net_benefit:,.0f}",
                delta=f"Rs.{net_benefit:,.0f}",
                delta_color="normal" if net_benefit > 0 else "inverse")
roi_col3.metric("ROI", f"{roi_pct:.1f}%",
                delta="Profitable" if roi_pct > 0 else "Loss-making",
                delta_color="normal" if roi_pct > 0 else "inverse")

# ROI signal box
if roi_pct > 50:
    st.markdown(f"""
    <div class="roi-positive">
        CAMPAIGN RECOMMENDED — Excellent ROI of {roi_pct:.1f}%!
        Expected net benefit: Rs.{net_benefit:,.0f} on a Rs.{total_cost:,.0f} investment.
    </div>
    """, unsafe_allow_html=True)
elif roi_pct > 0:
    st.markdown(f"""
    <div class="roi-neutral">
        MARGINAL — Positive ROI of {roi_pct:.1f}%, but consider if operational costs make it viable.
        Net benefit: Rs.{net_benefit:,.0f}.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="roi-negative">
        NOT RECOMMENDED — Negative ROI of {roi_pct:.1f}%.
        Consider reducing discount amount, targeting fewer customers, or improving targeting precision.
    </div>
    """, unsafe_allow_html=True)

# ─── Sensitivity Analysis ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Sensitivity Analysis — Discount vs ROI</div>', unsafe_allow_html=True)

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
    x=alt.X("Discount (Rs.):Q", title="Discount Amount per Customer (Rs.)"),
    y=alt.Y("ROI (%):Q", title="ROI (%)"),
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
st.caption(f"Yellow dashed line = current discount (Rs.{discount_amount}). Red dashed = break-even.")

# ─── Campaign Budget Optimizer ────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Budget Optimization — Find the Sweet Spot</div>',
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
            x=alt.X("Budget (Rs.):Q", title="Campaign Budget (Rs.)"),
            y=alt.Y("Net Benefit (Rs.):Q", title="Net Benefit (Rs.)"),
            tooltip=["Budget (Rs.)", "Net Benefit (Rs.)", "Expected Retained"]
        ).properties(height=280, title="Net Benefit vs Campaign Budget")
        st.altair_chart(net_chart, use_container_width=True)
    with col2:
        roi_opt_chart = alt.Chart(opt_df).mark_line(color="#f6e05e", strokeWidth=2.5).encode(
            x=alt.X("Budget (Rs.):Q", title="Campaign Budget (Rs.)"),
            y=alt.Y("ROI (%):Q", title="ROI (%)"),
            tooltip=["Budget (Rs.)", "ROI (%)"]
        ).properties(height=280, title="ROI vs Campaign Budget")
        zero_r = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            color="red", strokeDash=[4,4]
        ).encode(y="y:Q")
        st.altair_chart(roi_opt_chart + zero_r, use_container_width=True)

# ─── Target Customer Table ────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Targeted Customer List</div>', unsafe_allow_html=True)

show_cols = ["customerID","churn_probability","uplift_score","clv_estimate",
             "survival_months","customer_segment","MonthlyCharges","Contract"]
show_cols = [c for c in show_cols if c in target_df.columns]

st.markdown(f"**{n_targeted} customers selected for campaign:**")
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
    "Download Target Customer List (CSV)",
    csv, "roi_campaign_targets.csv", "text/csv"
)

# ─── Log Campaign ─────────────────────────────────────────────────────────────
if st.button("Save Campaign Simulation to Database", type="primary"):
    if DB_AVAILABLE:
        try:
            engine = get_engine()
            log_campaign(engine, total_discount_per_customer, n_targeted,
                         int(expected_retained), benefit)
            st.success(f"Campaign saved! ROI={roi_pct:.1f}%")
        except Exception as e:
            st.error(f"Could not save: {e}")
    else:
        st.info("Database logging not available. Results shown above.")

