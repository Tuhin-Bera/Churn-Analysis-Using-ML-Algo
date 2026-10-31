"""
LightGBM Churn Classifier + SHAP Explainer
============================================

This module trains and evaluates the primary customer churn classifier using LightGBM
and interprets its predictions using SHAP (SHapley Additive exPlanations).

Model Selection Rationale:
We selected LightGBM (Light Gradient Boosting Machine) for the following reasons:
1. Computational Efficiency: Uses histogram-based binning and leaf-wise tree growth,
   drastically reducing training latency compared to traditional XGBoost or Random Forests.
2. Imbalance Handling: The IBM Telco dataset exhibits an imbalanced target (~26.5% churn).
   Configuring 'scale_pos_weight' penalizes false negatives on churners proportionally.
3. Early Stopping Support: Prevents overfitting on tabular features by halting tree growth
   when validation metric ceases to improve.

Explainability Rationale:
In enterprise decision-making, black-box predictions are rarely adopted by marketing leadership.
Using SHAP TreeExplainer (derived from game-theoretic Shapley values), we compute exact marginal
feature attributions for every individual customer prediction, enabling marketing teams to answer
"WHY is this specific customer at risk?" before taking retention action.
"""

import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")   # Use non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")

# Define target artifact output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# LightGBM Classifier Training
# -----------------------------------------------------------------------------

def train_churn_model(X_train, y_train, X_val=None, y_val=None):
    """
    Trains a binary LightGBM gradient boosted decision tree classifier.
    
    Hyperparameter Logic:
    - objective='binary': Log-loss objective for probability calibration.
    - metric='auc': Area Under the ROC Curve ensures threshold-independent ranking quality.
    - scale_pos_weight: Ratio of negative to positive samples (~2.76x).
      This forces the algorithm to penalize missed churners heavily, boosting recall.
    - learning_rate=0.05 & n_estimators=500: Conservative step size paired with early stopping
      ensures optimal convergence without overfitting.
    - num_leaves=31 & min_child_samples=20: Regularizes tree complexity.
    """
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos  # ~2.76 for 26.5% churn rate

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,   # Subsamples 80% of features per tree split
        "bagging_fraction": 0.8,   # Subsamples 80% of rows to reduce variance
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)

    # If validation data is available, apply early stopping after 50 non-improving rounds
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1)
            ]
        )
    else:
        model.fit(X_train, y_train)

    print(f"[Model] LightGBM trained successfully with {model.n_estimators_} trees.")
    return model


# -----------------------------------------------------------------------------
# Comprehensive Model Evaluation
# -----------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, save_plots: bool = True) -> dict:
    """
    Generates full classification diagnostics across multiple evaluation metrics:
    - Accuracy: Overall correct predictions across both classes.
    - Precision: Of all predicted churners, how many actually churned?
    - Recall (Sensitivity): Of all true churners, how many did we catch? (Priority metric)
    - F1-Score: Harmonic balance between precision and recall.
    - ROC-AUC: Probability that a randomly chosen churner ranks higher than a stayer.
    
    Also generates and saves visual confusion matrices and ROC curves.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    print("\n[Model] === Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\n[Model] Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

    if save_plots:
        _plot_confusion_matrix(y_test, y_pred)
        _plot_roc_curve(model, X_test, y_test)
        _plot_feature_importance(model, X_test.columns.tolist())

    return metrics


def _plot_confusion_matrix(y_test, y_pred):
    """Generates and saves a clean confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Stay", "Predicted Churn"],
                yticklabels=["Actual Stay", "Actual Churn"], ax=ax)
    ax.set_title("Confusion Matrix — Churn Classifier", fontsize=13, pad=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved confusion matrix -> {path}")


def _plot_roc_curve(model, X_test, y_test):
    """Generates and saves the Receiver Operating Characteristic (ROC) curve."""
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name="LightGBM")
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)")
    ax.set_title("ROC Curve — Churn Classifier", fontsize=13, pad=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved ROC curve -> {path}")


def _plot_feature_importance(model, feature_names: list):
    """Generates and saves a horizontal bar chart of top feature split gains."""
    importance = pd.Series(model.feature_importances_, index=feature_names)
    top20 = importance.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    top20.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 20 Feature Importances (LightGBM Split Gain)", fontsize=13, pad=12)
    ax.set_xlabel("Relative Importance Score")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved feature importance -> {path}")


# -----------------------------------------------------------------------------
# SHAP Model Explainability
# -----------------------------------------------------------------------------

def compute_shap_values(model, X_background, X_explain=None):
    """
    Computes game-theoretic SHAP values via TreeExplainer.
    
    Why TreeExplainer:
    TreeExplainer computes exact Shapley values in polynomial time rather than
    exponential sampling time, making it suitable for production gradient boosting trees.
    """
    if X_explain is None:
        X_explain = X_background

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # Binary LightGBM returns either an ndarray or a 2-element list for classes [0, 1]
    # We extract class 1 (churn probability attributions)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    print(f"[SHAP] Computed SHAP values for {len(X_explain)} samples.")
    return explainer, shap_vals


def plot_shap_summary(explainer, shap_vals, X_explain, save: bool = True):
    """
    Generates a SHAP beeswarm summary plot.
    
    Interpretation:
    - Each dot is a customer observation.
    - Position on x-axis indicates whether the feature increased or decreased churn risk.
    - Color indicates feature value (red = high value, blue = low value).
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_explain, show=False)
    plt.title("SHAP Summary — Feature Impact on Churn Probability", fontsize=13)
    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[SHAP] Saved summary plot -> {path}")
    plt.close()


def plot_shap_bar(shap_vals, X_explain, save: bool = True):
    """Generates a clean bar plot ranking features by mean absolute SHAP attribution."""
    mean_abs = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=X_explain.columns
    ).nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    mean_abs.plot(kind="barh", ax=ax, color="coral")
    ax.set_title("Mean |SHAP| Values — Top 15 Features", fontsize=13, pad=12)
    ax.set_xlabel("Mean Absolute SHAP Attribution")
    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "shap_bar.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[SHAP] Saved bar plot -> {path}")
    plt.close()


def explain_single_customer(explainer, X_row: pd.DataFrame):
    """
    Extracts local feature attributions for an individual customer.
    Powers the 'Why will this customer leave?' view on the Streamlit dashboard.
    """
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv = sv[1]
    contributions = pd.DataFrame({
        "feature": X_row.columns,
        "value": X_row.values[0],
        "shap_contribution": sv[0]
    }).sort_values("shap_contribution", key=abs, ascending=False)
    return contributions


# -----------------------------------------------------------------------------
# Model Persistence & Artifact Serialization
# -----------------------------------------------------------------------------

def save_model(model, explainer, path: str = None):
    """Serializes the trained LightGBM model and SHAP explainer to disk."""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "lgbm_churn_model.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "explainer": explainer}, f)
    print(f"[Model] Saved model bundle to {path}")
    return path


def load_model(path: str = None):
    """Loads a previously serialized model and explainer bundle."""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "lgbm_churn_model.pkl")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"[Model] Loaded model bundle from {path}")
    return bundle["model"], bundle["explainer"]
