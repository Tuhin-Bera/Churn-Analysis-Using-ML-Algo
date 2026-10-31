"""
Page 2: Individual Churn Predictor with SHAP
==============================================

This page lets business users look up any customer and see:
1. Their churn probability (with risk level)
2. WHY they're predicted to churn (SHAP explanation)
3. Their predicted remaining tenure and CLV
4. Recommended action
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.feature_engineering import preprocess_for_model, load_raw_data, engineer_features

st.set_page_config(
    page_title="Churn Predictor | Churn Intelligence",
    page_icon="🔮",
    layout="wide"
)

st.markdown("""
<style>
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #667eea;
        border-bottom: 2px solid #667eea; padding-bottom: 6px; margin: 20px 0 14px;
    }
    .risk-high { background: #fed7d7; color: #c53030; padding: 8px 16px; border-radius: 8px;
                 font-weight: 700; font-size: 1.1rem; display: inline-block; }
    .risk-med  { background: #feebc8; color: #c05621; padding: 8px 16px; border-radius: 8px;
                 font-weight: 700; font-size: 1.1rem; display: inline-block; }
    .risk-low  { background: #c6f6d5; color: #276749; padding: 8px 16px; border-radius: 8px;
                 font-weight: 700; font-size: 1.1rem; display: inline-block; }
    .customer-card {
        background: #1e2130; border-radius: 12px; padding: 20px;
        border: 1px solid #3d4460; margin: 10px 0;
    }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #2a2f45);
        border-radius: 12px; padding: 12px 16px; border: 1px solid #3d4460;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Individual Churn Predictor")
st.caption("Look up any customer to see their churn risk, explanation, and recommendations.")


# ─── Load data and models ─────────────────────────────────────────────────────
@st.cache_data
def load_master():
    path = os.path.join(ROOT, "outputs", "master_predictions.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_resource
def load_model_and_explainer():
    path = os.path.join(ROOT, "outputs", "lgbm_churn_model.pkl")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["explainer"]

@st.cache_data
def load_raw():
    path = os.path.join(ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return load_raw_data(path) if os.path.exists(path) else None

@st.cache_data
def get_feature_cols():
    meta_path = os.path.join(ROOT, "outputs", "pipeline_meta.pkl")
    if not os.path.exists(meta_path):
        return []
    with open(meta_path, "rb") as f:
        return pickle.load(f).get("feature_cols", [])

df_master = load_master()
model, explainer = load_model_and_explainer()
df_raw = load_raw()
feature_cols = get_feature_cols()

if df_master is None or model is None:
    st.error("Run `python train_pipeline.py` first.")
    st.stop()

# ─── Customer Search ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Customer Lookup</div>', unsafe_allow_html=True)

search_method = st.radio(
    "Select customer by:",
    ["Customer ID", "Browse High-Risk Customers", "Manual Input (New Customer)"],
    horizontal=True
)

selected_customer_id = None

if search_method == "Customer ID":
    customer_id = st.selectbox(
        "Search Customer ID:",
        options=df_master["customerID"].tolist(),
        index=0
    )
    selected_customer_id = customer_id

elif search_method == "Browse High-Risk Customers":
    high_risk = df_master[df_master["churn_probability"] >= 0.65].sort_values(
        "churn_probability", ascending=False
    )[["customerID","churn_probability","clv_estimate","customer_segment","Contract","tenure"]].head(50)

    st.dataframe(
        high_risk.style.format({
            "churn_probability": "{:.1%}",
            "clv_estimate": "Rs.{:,.0f}"
        }),
        use_container_width=True, hide_index=True
    )
    selected_customer_id = st.selectbox(
        "Select a customer to analyze:", high_risk["customerID"].tolist()
    )

elif search_method == "Manual Input (New Customer)":
    st.markdown("**Enter customer details to predict churn:**")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        tenure_in = st.slider("Tenure (months)", 0, 72, 12)
        monthly_in = st.slider("Monthly Charges", 18.0, 120.0, 65.0)
        contract_in = st.selectbox("Contract Type",
            ["Month-to-month", "One year", "Two year"])
    with mc2:
        internet_in = st.selectbox("Internet Service",
            ["DSL", "Fiber optic", "No"])
        payment_in = st.selectbox("Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"])
        senior_in = st.selectbox("Senior Citizen", ["No", "Yes"])
    with mc3:
        partner_in = st.selectbox("Has Partner", ["Yes", "No"])
        phone_in = st.selectbox("Phone Service", ["Yes", "No"])
        security_in = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

    if st.button("Predict Churn for New Customer", type="primary"):
        # Build a fake row
        total_charges = tenure_in * monthly_in
        contract_risk_map = {"Month-to-month": 3, "One year": 2, "Two year": 1}
        cr = contract_risk_map[contract_in]
        svc = sum([
            phone_in == "Yes",
            internet_in in ["DSL","Fiber optic"],
            security_in == "Yes",
        ])
        avg_monthly = total_charges / (tenure_in + 1) if tenure_in > 0 else monthly_in
        charge_ratio = monthly_in / (avg_monthly + 1e-6)

        manual_X = pd.DataFrame([{
            "tenure": tenure_in,
            "MonthlyCharges": monthly_in,
            "TotalCharges": total_charges,
            "avg_monthly_charge": avg_monthly,
            "charge_ratio": charge_ratio,
            "service_count": svc,
            "contract_risk": cr,
            "is_senior": 1 if senior_in == "Yes" else 0,
            "is_electronic_check": 1 if payment_in == "Electronic check" else 0,
            "has_internet": 1 if internet_in != "No" else 0,
            "paperless": 1,
            "has_family": 1 if partner_in == "Yes" else 0,
            "is_high_spender": 1 if monthly_in > 65 else 0,
            "gender_Male": 0,
            "InternetService_Fiber optic": 1 if internet_in == "Fiber optic" else 0,
            "InternetService_No": 1 if internet_in == "No" else 0,
            "gender_Female": 0,
            "InternetService_DSL": 1 if internet_in == "DSL" else 0,
        }])

        # Align with feature_cols
        for col in feature_cols:
            if col not in manual_X.columns:
                manual_X[col] = 0
        manual_X = manual_X[feature_cols].fillna(0)

        prob = model.predict_proba(manual_X)[0, 1]
        st.markdown("---")
        risk_level = "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW")
        risk_class = {"HIGH": "risk-high", "MEDIUM": "risk-med", "LOW": "risk-low"}[risk_level]
        st.markdown(f"""
        **Predicted Churn Probability:** 
        <span class="{risk_class}">{prob:.1%} ({risk_level} RISK)</span>
        """, unsafe_allow_html=True)

        # SHAP for manual customer
        if explainer is not None:
            sv = explainer.shap_values(manual_X)
            sv = sv[1] if isinstance(sv, list) else sv
            contrib = pd.DataFrame({
                "Feature": manual_X.columns,
                "Value": manual_X.values[0],
                "SHAP Impact": sv[0]
            }).sort_values("SHAP Impact", key=abs, ascending=True).tail(10)

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#fc8181" if v > 0 else "#68d391" for v in contrib["SHAP Impact"]]
            ax.barh(contrib["Feature"], contrib["SHAP Impact"], color=colors)
            ax.axvline(0, color="white", linewidth=0.8)
            ax.set_title("SHAP Feature Contributions (New Customer)", fontsize=12, color="white")
            ax.set_facecolor("#1e2130")
            fig.patch.set_facecolor("#1e2130")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.stop()

# ─── Show Selected Customer Analysis ─────────────────────────────────────────
if selected_customer_id:
    row = df_master[df_master["customerID"] == selected_customer_id].iloc[0]
    raw_row = df_raw[df_raw["customerID"] == selected_customer_id].iloc[0]

    st.markdown("---")
    st.markdown(f'<div class="section-header">Customer Profile: {selected_customer_id}</div>',
                unsafe_allow_html=True)

    # Risk badge
    prob = row["churn_probability"]
    risk_level = "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW")
    risk_class = {"HIGH": "risk-high", "MEDIUM": "risk-med", "LOW": "risk-low"}[risk_level]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Churn Probability", f"{prob:.1%}")
    c2.metric("Risk Level", risk_level)
    c3.metric("CLV Estimate", f"Rs.{row['clv_estimate']:,.0f}")
    c4.metric("Predicted Survival", f"{row['survival_months']:.0f} months")
    c5.metric("Segment", row["customer_segment"])

    # Customer details
    st.markdown("**Customer Details:**")
    detail_cols = ["gender", "SeniorCitizen", "Partner", "Dependents",
                   "tenure", "Contract", "InternetService", "MonthlyCharges",
                   "TotalCharges", "PaymentMethod", "Churn"]
    detail_cols = [c for c in detail_cols if c in df_master.columns]
    details = df_master[df_master["customerID"] == selected_customer_id][detail_cols]
    st.dataframe(details, use_container_width=True, hide_index=True)

    # ─── SHAP Explanation ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Why Is This Customer at Risk? (SHAP)</div>',
                unsafe_allow_html=True)

    # Get features for this customer
    X_all, y_all, feat_cols = preprocess_for_model(df_raw)
    cust_idx = df_raw[df_raw["customerID"] == selected_customer_id].index[0]
    X_cust = X_all.iloc[[cust_idx]]

    if explainer is not None:
        sv = explainer.shap_values(X_cust)
        sv = sv[1] if isinstance(sv, list) else sv

        contrib = pd.DataFrame({
            "Feature": X_cust.columns,
            "Value": X_cust.values[0],
            "SHAP Impact": sv[0]
        }).sort_values("SHAP Impact", key=abs, ascending=True).tail(12)

        fig, ax = plt.subplots(figsize=(9, 6))
        colors_list = ["#fc8181" if v > 0 else "#68d391" for v in contrib["SHAP Impact"]]
        bars = ax.barh(contrib["Feature"], contrib["SHAP Impact"], color=colors_list)

        # Add value labels
        for bar_val, val, feat_val in zip(contrib["SHAP Impact"], contrib["Value"], contrib["Feature"]):
            ax.text(bar_val + (0.005 if bar_val > 0 else -0.005),
                    feat_val, f"  val={val:.2f}", va="center",
                    color="white", fontsize=7)

        ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value (positive = increases churn risk)", color="white")
        ax.set_title(f"SHAP Waterfall — Why {selected_customer_id} is {risk_level} Risk",
                     fontsize=12, color="white")
        ax.set_facecolor("#1e2130")
        fig.patch.set_facecolor("#1e2130")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Text explanation
        top3 = contrib.sort_values("SHAP Impact", ascending=False).head(3)
        st.markdown("**Top 3 churn drivers for this customer:**")
        for _, r2 in top3.iterrows():
            direction = "increases" if r2["SHAP Impact"] > 0 else "decreases"
            st.markdown(f"- **{r2['Feature']}** = {r2['Value']:.2f} → {direction} churn risk by {abs(r2['SHAP Impact']):.3f}")

    # ─── Recommended Action ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Recommended Retention Action</div>',
                unsafe_allow_html=True)

    segment = row["customer_segment"]
    uplift_cat = row.get("uplift_category", "Unknown")
    clv = row["clv_estimate"]

    if risk_level == "HIGH" and uplift_cat == "Persuadable":
        st.success(f"""
        **Priority Action: Proactive Retention Offer**
        
        This customer is HIGH RISK and a **Persuadable** — meaning they are likely to respond
        to a retention offer. Their CLV of Rs.{clv:,.0f} justifies a discount of up to Rs.{clv*0.05:,.0f}.
        
        Recommended: Call with a loyalty discount or contract upgrade offer within 7 days.
        """)
    elif risk_level == "HIGH" and uplift_cat == "Lost Cause":
        st.error(f"""
        **Monitor Only — Low Response Likelihood**
        
        This customer is HIGH RISK but classified as a **Lost Cause** — offers are unlikely to change
        their behavior. Their CLV of Rs.{clv:,.0f} may not justify a large discount spend.
        
        Recommended: Ensure service quality; do not invest heavily in win-back campaign.
        """)
    elif risk_level == "LOW":
        st.info(f"""
        **No Action Needed**
        
        This customer has LOW churn risk. They're in the **{segment}** segment with
        CLV of Rs.{clv:,.0f}. Focus on maintaining service satisfaction.
        """)
    else:
        st.warning(f"""
        **Monitor — Medium Risk**
        
        This customer is at MEDIUM risk. Consider a proactive check-in call.
        Segment: **{segment}** | CLV: Rs.{clv:,.0f} | Uplift Category: {uplift_cat}
        """)

