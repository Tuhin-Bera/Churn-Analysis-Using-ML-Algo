"""
Master Training Pipeline
=========================

This is the central orchestration script for the Customer Lifetime Value & Churn Intelligence System.
It executes the end-to-end data science and machine learning workflow sequentially:

  Step 1: Database Setup & Data Ingestion (PostgreSQL / SQLite fallback)
  Step 2: Data Cleaning & Domain Feature Engineering
  Step 3: LightGBM Churn Classifier Training, Evaluation & SHAP Explainability
  Step 4: Survival Analysis (Kaplan-Meier Survival Curves & Cox Proportional Hazards)
  Step 5: Uplift Modeling (CausalML S-Learner for Incremental Offer Response)
  Step 6: Customer Lifetime Value (DCF Annuity) & K-Means Customer Clustering (k=4)
  Step 7: Database Persistence & Artifact Export for the Streamlit Application

Execution:
    python train_pipeline.py

All trained model weights, explainers, figures, and scored datasets are written to 'outputs/'.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Load .env variables so DATABASE_URL and PG_* credentials are available
# when running locally or in CI before Streamlit starts.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure project root is available on sys.path regardless of execution directory
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Internal module imports
from src.database.db_setup import get_engine, create_tables, load_csv_to_db, save_predictions, save_segments
from src.utils.feature_engineering import (
    load_raw_data, preprocess_for_model, split_data,
    prepare_survival_data, compute_clv
)
from src.models.churn_model import (
    train_churn_model, evaluate_model,
    compute_shap_values, plot_shap_summary, plot_shap_bar,
    save_model
)
from src.models.survival_model import (
    plot_km_overall, plot_km_by_contract, plot_km_by_internet,
    fit_cox_model, plot_cox_coefficients, predict_survival_months
)
from src.models.uplift_model import (
    simulate_treatment_assignment, simulate_outcome_with_treatment,
    compute_uplift_scores, categorize_uplift, plot_uplift_distribution
)
from src.models.clv_segmentation import (
    compute_clv_detailed, plot_clv_distribution,
    segment_customers, plot_segments, get_segment_summary
)

OUTPUT_DIR = os.path.join(ROOT, "outputs")
DATA_PATH = os.path.join(ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


def banner(msg: str):
    """Prints a structured section header in the terminal logs."""
    print("\n" + "=" * 65)
    print(f"  {msg}")
    print("=" * 65)


def main():
    start_time = time.time()

    # -------------------------------------------------------------------------
    # STEP 1: Relational Database Initialization & Ingestion
    # -------------------------------------------------------------------------
    banner("STEP 1 -- Database Setup & Data Ingestion")
    # Connect to PostgreSQL, or fall back to local SQLite if unavailable
    engine = get_engine(use_sqlite_fallback=True)
    create_tables(engine)
    
    # Ingest the 7,043 raw customer records from CSV into the database
    n_loaded = load_csv_to_db(DATA_PATH, engine, if_exists="replace")
    print(f"[OK] {n_loaded} customer records loaded into relational database.")

    # Load and sanitize raw dataframe in memory for modeling
    banner("Loading & Sanitizing Dataset")
    df_raw = load_raw_data(DATA_PATH)
    print(f"[OK] Dataset shape: {df_raw.shape}")
    print(f"  Baseline churn rate: {df_raw['Churn_binary'].mean():.1%}")

    # -------------------------------------------------------------------------
    # STEP 2 & 3: Feature Engineering & LightGBM Churn Classifier
    # -------------------------------------------------------------------------
    banner("STEP 2 & 3 -- Feature Engineering & Churn Classifier (LightGBM)")
    # Generate domain features and one-hot encodings
    X, y, feature_cols = preprocess_for_model(df_raw)
    
    # Stratified split ensures exact class distribution matching in train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"[OK] Train split: {len(X_train):,} rows | Test split: {len(X_test):,} rows")
    print(f"  Feature dimensions: {len(feature_cols)} columns")

    # Fit LightGBM classifier with early stopping on validation split
    model = train_churn_model(X_train, y_train, X_test, y_test)
    metrics = evaluate_model(model, X_test, y_test, save_plots=True)

    # -------------------------------------------------------------------------
    # STEP 3 (cont): SHAP Model Explainability
    # -------------------------------------------------------------------------
    banner("STEP 3 (cont) -- SHAP Explainability & Global Importance")
    # Subsample 300 background samples for TreeExplainer computation
    X_sample = X_train.sample(min(300, len(X_train)), random_state=42)
    explainer, shap_vals = compute_shap_values(model, X_sample, X_sample)
    plot_shap_summary(explainer, shap_vals, X_sample)
    plot_shap_bar(shap_vals, X_sample)

    # Serialize model and explainer bundle to outputs/
    model_path = save_model(model, explainer)

    # Score full customer base to get calibrated churn probabilities
    X_all, y_all, _ = preprocess_for_model(df_raw)
    churn_probs_all = pd.Series(
        model.predict_proba(X_all)[:, 1],
        name="churn_probability"
    )
    print(f"[OK] Churn probabilities computed for all {len(churn_probs_all):,} customers.")

    # -------------------------------------------------------------------------
    # STEP 4: Lifelines Customer Survival Analysis
    # -------------------------------------------------------------------------
    banner("STEP 4 -- Survival Analysis (Kaplan-Meier + Cox PH)")
    survival_df = prepare_survival_data(df_raw)

    # Non-parametric Kaplan-Meier curves
    kmf_overall = plot_km_overall(survival_df)
    kmf_contract = plot_km_by_contract(survival_df)
    plot_km_by_internet(survival_df)

    # Semi-parametric Cox Proportional Hazards regression
    cph = fit_cox_model(survival_df)
    plot_cox_coefficients(cph)

    # Predict continuous expected survival lifetime in months
    pred_survival = predict_survival_months(cph, survival_df)
    print(f"[OK] Survival predictions completed: Median expected survival = {pred_survival.median():.1f} months")

    # -------------------------------------------------------------------------
    # STEP 5: Uplift Modeling with CausalML
    # -------------------------------------------------------------------------
    banner("STEP 5 -- Uplift Modeling (CausalML S-Learner)")
    # Simulate historical marketing retention offer assignment
    treatment = simulate_treatment_assignment(churn_probs_all)
    outcome = simulate_outcome_with_treatment(
        df_raw["Churn_binary"], churn_probs_all, treatment, lift_effect=0.20
    )
    
    # Compute Individual Treatment Effects (ITE) and categorize into 4 quadrants
    uplift_scores = compute_uplift_scores(X_all, treatment, outcome)
    uplift_cats = categorize_uplift(uplift_scores, churn_probs_all)
    plot_uplift_distribution(uplift_scores, uplift_cats)
    print(f"[OK] Uplift categorization: {uplift_cats.value_counts().to_dict()}")

    # -------------------------------------------------------------------------
    # STEP 6: Customer Lifetime Value (CLV) & K-Means Segmentation
    # -------------------------------------------------------------------------
    banner("STEP 6 -- Customer Lifetime Value (CLV) & Behavioral Clustering")
    # Discounted Cash Flow CLV using predicted survival months from Step 4
    clv_df = compute_clv_detailed(df_raw, pred_survival)
    plot_clv_distribution(clv_df)
    print(f"[OK] CLV Statistics: Mean = Rs.{clv_df['clv_estimate'].mean():,.0f} | Max = Rs.{clv_df['clv_estimate'].max():,.0f}")

    # Unsupervised K-Means clustering (k=4)
    segment_df, km_model, scaler, seg_feature_cols = segment_customers(
        df_raw, churn_probs_all, clv_df["clv_estimate"], n_clusters=4
    )
    plot_segments(segment_df)
    segment_summary = get_segment_summary(segment_df)
    print(f"\n[Segments Overview]\n{segment_summary[['customer_count','avg_churn_prob','avg_clv']].to_string()}")

    # -------------------------------------------------------------------------
    # STEP 7: Relational Database Persistence & Master Export
    # -------------------------------------------------------------------------
    banner("Saving Model Predictions & Segments to Database")
    # Upsert scored customer records into 'predictions' table
    predictions_df = pd.DataFrame({
        "customer_id": df_raw["customerID"].values,
        "churn_probability": churn_probs_all.values,
        "churn_prediction": (churn_probs_all >= 0.5).astype(int).values,
        "clv_estimate": clv_df["clv_estimate"].values,
        "survival_months": pred_survival.values,
        "uplift_score": uplift_scores.values,
        "uplift_category": uplift_cats.values,
        "customer_segment": segment_df["segment"].values,
    })
    save_predictions(predictions_df, engine)

    # Upsert segment metrics into 'segments' table
    segment_summary_db = segment_summary.reset_index()
    segment_summary_db.columns = [
        "segment_name", "customer_count", "avg_tenure",
        "avg_monthly_charges", "avg_churn_prob", "avg_clv", "strategy"
    ]
    segment_summary_db["segment_id"] = range(len(segment_summary_db))
    segment_summary_db["churn_rate"] = segment_summary_db["avg_churn_prob"]
    save_segments(segment_summary_db, engine)

    # Export consolidated master CSV for high-speed Streamlit cache loading
    banner("Exporting Consolidated Master Dataset")
    master_df = df_raw.copy()
    master_df["churn_probability"] = churn_probs_all.values
    master_df["churn_prediction"] = (churn_probs_all >= 0.5).astype(int).values
    master_df["clv_estimate"] = clv_df["clv_estimate"].values
    master_df["survival_months"] = pred_survival.values
    master_df["uplift_score"] = uplift_scores.values
    master_df["uplift_category"] = uplift_cats.values
    master_df["customer_segment"] = segment_df["segment"].values

    master_path = os.path.join(OUTPUT_DIR, "master_predictions.csv")
    master_df.to_csv(master_path, index=False)
    print(f"[OK] Master predictions saved -> {master_path}")

    # Save pipeline metadata
    meta = {
        "feature_cols": feature_cols,
        "seg_feature_cols": seg_feature_cols,
        "metrics": metrics,
    }
    meta_path = os.path.join(OUTPUT_DIR, "pipeline_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"[OK] Pipeline metadata saved -> {meta_path}")

    # Pipeline Completion Summary
    elapsed = time.time() - start_time
    banner(f"MASTER PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f}s")
    print("  Ready to launch the interactive dashboard:")
    print("    streamlit run app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
