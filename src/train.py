"""
src/train.py
============
Model Training Module.

Trains three classifiers and selects the best one:
  1. Logistic Regression       - baseline / explainability
  2. Random Forest             - robust ensemble
  3. XGBoost                   - state-of-the-art gradient boosting

Evaluation metric: ROC-AUC (best for imbalanced data)
Also reports: Precision, Recall, F1, Average Precision

Author  : Fraud Detection System
Version : 1.0
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    classification_report,
)
import xgboost as xgb
import joblib

# ----------------------------------------------
# Paths
# ----------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


# ==============================================
# Model Definitions
# ==============================================

def _build_models() -> dict:
    """
    Return a dict of model_name -> sklearn-compatible estimator.
    All models are configured for fraud (imbalanced) scenarios.
    """
    return {
        "LogisticRegression": LogisticRegression(
            C            = 0.1,
            class_weight = "balanced",
            max_iter     = 500,
            solver       = "lbfgs",
            random_state = RANDOM_SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators = 200,
            max_depth    = 12,
            class_weight = "balanced_subsample",
            n_jobs       = -1,
            random_state = RANDOM_SEED,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators       = 300,
            max_depth          = 6,
            learning_rate      = 0.05,
            subsample          = 0.8,
            colsample_bytree   = 0.8,
            scale_pos_weight   = 1,          # SMOTE already handled balance
            eval_metric        = "aucpr",
            use_label_encoder  = False,
            random_state       = RANDOM_SEED,
            verbosity          = 0,
        ),
    }


# ==============================================
# Training & Evaluation
# ==============================================

def _evaluate(model, X, y, label: str = "") -> dict:
    """Return a metrics dict for a fitted model on (X, y)."""
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "label"             : label,
        "ROC_AUC"           : round(roc_auc_score(y, y_proba),            4),
        "Avg_Precision"     : round(average_precision_score(y, y_proba),   4),
        "Precision"         : round(precision_score(y, y_pred,             zero_division=0), 4),
        "Recall"            : round(recall_score(y, y_pred,                zero_division=0), 4),
        "F1"                : round(f1_score(y, y_pred,                    zero_division=0), 4),
    }


def train_all_models(
    X_train, y_train,
    X_val,   y_val,
) -> tuple[dict, str]:
    """
    Train all models, evaluate on validation set.

    Returns
    -------
    (results_dict, best_model_name)
    where results_dict = { model_name: {"model": ..., "val_metrics": {...}} }
    """
    models  = _build_models()
    results = {}

    for name, model in models.items():
        print(f"\n[Train] -- Training {name} ----------------------")
        model.fit(X_train, y_train)

        train_metrics = _evaluate(model, X_train, y_train, "train")
        val_metrics   = _evaluate(model, X_val,   y_val,   "val")

        results[name] = {
            "model"        : model,
            "train_metrics": train_metrics,
            "val_metrics"  : val_metrics,
        }

        print(f"  Train ROC-AUC : {train_metrics['ROC_AUC']}")
        print(f"  Val   ROC-AUC : {val_metrics['ROC_AUC']}")
        print(f"  Val   Recall  : {val_metrics['Recall']}")
        print(f"  Val   F1      : {val_metrics['F1']}")

    # -- Pick best by validation ROC-AUC ------
    best_name = max(results, key=lambda k: results[k]["val_metrics"]["ROC_AUC"])
    print(f"\n[Train] [OK] Best model: {best_name}  "
          f"(Val ROC-AUC = {results[best_name]['val_metrics']['ROC_AUC']})")

    return results, best_name


def save_best_model(results: dict, best_name: str, feature_names: list = None) -> Path:
    """Persist the best model and feature names to disk."""
    model_path = MODEL_DIR / "best_model.pkl"
    joblib.dump(results[best_name]["model"], model_path)
    print(f"[Train] Best model saved -> {model_path}")
    if feature_names is not None:
        feat_path = MODEL_DIR / "feature_names.pkl"
        joblib.dump(feature_names, feat_path)
        print(f"[Train] Feature names saved -> {feat_path}")
    return model_path


def load_model(path: str | None = None):
    """Load a saved model from disk."""
    p = Path(path) if path else MODEL_DIR / "best_model.pkl"
    model = joblib.load(p)
    print(f"[Train] Model loaded from {p}")
    return model


# ----------------------------------------------
if __name__ == "__main__":
    # Quick smoke-test with dummy data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]
    res, best = train_all_models(X_train, y_train, X_val, y_val)
    save_best_model(res, best)
