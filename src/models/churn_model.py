"""
LightGBM Churn Classifier + SHAP Explainer
============================================

This module trains the main churn prediction model.
We chose LightGBM over XGBoost because:
  1. Faster training on this size dataset
  2. Better handling of categorical features
  3. Lower memory usage

SHAP (SHapley Additive exPlanations) is used to understand WHY the model
predicts what it predicts — crucial for business stakeholders who need
justification before acting on model outputs.
"""

import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
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

# Output directory for saved artifacts
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- model training -----------------------------------------------------------

def train_churn_model(X_train, y_train, X_val=None, y_val=None):
    """
    Train a LightGBM binary classifier.
    
    Hyperparameters were chosen after quick experimentation.
    scale_pos_weight handles the class imbalance (≈26% churn rate).
    """
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos  # upweight minority (churn) class

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)]
        )
    else:
        model.fit(X_train, y_train)

    print(f"[Model] LightGBM trained with {model.n_estimators_} trees.")
    return model


def evaluate_model(model, X_test, y_test, save_plots: bool = True) -> dict:
    """
    Full model evaluation with all the metrics the portal asks for.
    Also saves confusion matrix and ROC curve plots.
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

    print("\n[Model] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

    if save_plots:
        _plot_confusion_matrix(y_test, y_pred)
        _plot_roc_curve(model, X_test, y_test)
        _plot_feature_importance(model, X_test.columns.tolist())

    return metrics


def _plot_confusion_matrix(y_test, y_pred):
    """Save confusion matrix heatmap."""
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
    """Save ROC curve."""
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name="LightGBM")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curve — Churn Classifier", fontsize=13, pad=12)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved ROC curve -> {path}")


def _plot_feature_importance(model, feature_names: list):
    """Bar chart of top-20 feature importances."""
    importance = pd.Series(model.feature_importances_, index=feature_names)
    top20 = importance.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    top20.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 20 Feature Importances (LightGBM Gain)", fontsize=13, pad=12)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved feature importance -> {path}")


# --- SHAP explainability ------------------------------------------------------

def compute_shap_values(model, X_background, X_explain=None):
    """
    Compute SHAP values using TreeExplainer (fastest for tree models).
    
    X_background: the training data (used to build the explainer)
    X_explain: data to explain — defaults to X_background (summary plots)
    """
    if X_explain is None:
        X_explain = X_background

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # For binary classification, LightGBM returns either a 2D array or
    # a list of 2 arrays — we want the positive class (index 1)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    print(f"[SHAP] Computed SHAP values for {len(X_explain)} samples.")
    return explainer, shap_vals


def plot_shap_summary(explainer, shap_vals, X_explain, save: bool = True):
    """Beeswarm summary plot — shows which features drive churn."""
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
    """Mean absolute SHAP values — simpler bar chart version."""
    mean_abs = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=X_explain.columns
    ).nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    mean_abs.plot(kind="barh", ax=ax, color="coral")
    ax.set_title("Mean |SHAP| Values — Top 15 Features", fontsize=13, pad=12)
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "shap_bar.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[SHAP] Saved bar plot -> {path}")
    plt.close()


def explain_single_customer(explainer, X_row: pd.DataFrame):
    """
    Returns a DataFrame of SHAP contributions for one customer.
    Useful for the Streamlit 'Why will this customer churn?' view.
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


# --- model persistence --------------------------------------------------------

def save_model(model, explainer, path: str = None):
    """Pickle the trained model and SHAP explainer."""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "lgbm_churn_model.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "explainer": explainer}, f)
    print(f"[Model] Saved model to {path}")
    return path


def load_model(path: str = None):
    """Load a previously trained model."""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "lgbm_churn_model.pkl")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"[Model] Loaded model from {path}")
    return bundle["model"], bundle["explainer"]
