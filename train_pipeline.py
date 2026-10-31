"""
Master Training Pipeline
=========================
Author: Tuhin Bera | Data Science Intern -- Nexvia Technologies
Date: October 2026

Run this script ONCE before launching the Streamlit app.
It will:
  1. Set up PostgreSQL (or SQLite) database
  2. Load the CSV into the database
  3. Train the LightGBM churn classifier
  4. Compute SHAP explanations
  5. Run survival analysis (Kaplan-Meier + Cox PH)
  6. Compute uplift scores with CausalML
  7. Calculate CLV for all customers
  8. Segment customers with K-Means
  9. Save all outputs (models + predictions DB table + plots)

Usage:
    python train_pipeline.py

All outputs go to the /outputs/ directory.
The Streamlit app reads from /outputs/ -- so run this first!
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Make sure local imports work regardless of where we run from
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

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


def banner(msg):
    """Print a nicely formatted section banner."""
    print("\n" + "=" * 65)
    print(f"  {msg}")
    print("=" * 65)


def main():
    start_time = time.time()

    # Step 1: Database Setup
    banner("STEP 1 -- Database Setup & Data Loading")
    engine = get_engine(use_sqlite_fallback=True)
    create_tables(engine)
    n_loaded = load_csv_to_db(DATA_PATH, engine, if_exists="replace")
    print(f"[OK] {n_loaded} customer records loaded into database.")

    # Load raw data
    banner("Loading Raw Dataset")
    df_raw = load_raw_data(DATA_PATH)
    print(f"[OK] Dataset shape: {df_raw.shape}")
    print(f"  Churn rate: {df_raw['Churn_binary'].mean():.1%}")

    # Step 2 & 3: Feature Engineering + Churn Model
    banner("STEP 2 & 3 -- Feature Engineering & Churn Classifier (LightGBM)")
    X, y, feature_cols = preprocess_for_model(df_raw)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"[OK] Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"  Features: {len(feature_cols)}")

    # Train LightGBM
    model = train_churn_model(X_train, y_train, X_test, y_test)
    metrics = evaluate_model(model, X_test, y_test, save_plots=True)

    # SHAP Explainability
    banner("STEP 3 -- SHAP Explainability")
    # Use a sample of 300 for SHAP (faster, still representative)
    X_sample = X_train.sample(min(300, len(X_train)), random_state=42)
    explainer, shap_vals = compute_shap_values(model, X_sample, X_sample)
    plot_shap_summary(explainer, shap_vals, X_sample)
    plot_shap_bar(shap_vals, X_sample)

    # Save model + explainer
    model_path = save_model(model, explainer)

    # Predict probabilities for ALL customers
    X_all, y_all, _ = preprocess_for_model(df_raw)
    churn_probs_all = pd.Series(
        model.predict_proba(X_all)[:, 1],
        name="churn_probability"
    )
    print(f"[OK] Churn probabilities computed for {len(churn_probs_all)} customers.")

    # Step 4: Survival Analysis
    banner("STEP 4 -- Survival Analysis (Kaplan-Meier + Cox PH)")
    survival_df = prepare_survival_data(df_raw)

    kmf_overall = plot_km_overall(survival_df)
    kmf_contract = plot_km_by_contract(survival_df)
    plot_km_by_internet(survival_df)

    cph = fit_cox_model(survival_df)
    plot_cox_coefficients(cph)

    # Predict survival months for each customer
    pred_survival = predict_survival_months(cph, survival_df)
    print(f"[OK] Survival predictions: median={pred_survival.median():.1f} months")

    # Step 5: Uplift Modeling
    banner("STEP 5 -- Uplift Modeling (CausalML)")
    treatment = simulate_treatment_assignment(churn_probs_all)
    outcome = simulate_outcome_with_treatment(
        df_raw["Churn_binary"], churn_probs_all, treatment, lift_effect=0.20
    )
    uplift_scores = compute_uplift_scores(X_all, treatment, outcome)
    uplift_cats = categorize_uplift(uplift_scores, churn_probs_all)
    plot_uplift_distribution(uplift_scores, uplift_cats)
    print(f"[OK] Uplift categories: {uplift_cats.value_counts().to_dict()}")

    # Step 6: CLV + Segmentation
    banner("STEP 6 -- CLV Computation & Customer Segmentation")
    clv_df = compute_clv_detailed(df_raw, pred_survival)
    plot_clv_distribution(clv_df)
    print(f"[OK] CLV: mean=Rs.{clv_df['clv_estimate'].mean():,.0f} | max=Rs.{clv_df['clv_estimate'].max():,.0f}")

    segment_df, km_model, scaler, seg_feature_cols = segment_customers(
        df_raw, churn_probs_all, clv_df["clv_estimate"], n_clusters=4
    )
    plot_segments(segment_df)
    segment_summary = get_segment_summary(segment_df)
    print(f"\n[Segments]\n{segment_summary[['customer_count','avg_churn_prob','avg_clv']].to_string()}")

    # Save all predictions to DB
    banner("Saving Predictions to Database")
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

    segment_summary_db = segment_summary.reset_index()
    segment_summary_db.columns = [
        "segment_name", "customer_count", "avg_tenure",
        "avg_monthly_charges", "avg_churn_prob", "avg_clv", "strategy"
    ]
    segment_summary_db["segment_id"] = range(len(segment_summary_db))
    segment_summary_db["churn_rate"] = segment_summary_db["avg_churn_prob"]
    save_segments(segment_summary_db, engine)

    # Save combined master DataFrame for Streamlit
    banner("Saving Master Output File")
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

    # Save the feature columns and metadata for the Streamlit app
    meta = {
        "feature_cols": feature_cols,
        "seg_feature_cols": seg_feature_cols,
        "metrics": metrics,
    }
    meta_path = os.path.join(OUTPUT_DIR, "pipeline_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"[OK] Pipeline metadata saved -> {meta_path}")

    # Done
    elapsed = time.time() - start_time
    banner(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print("  Run the Streamlit app with:")
    print("    streamlit run app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
