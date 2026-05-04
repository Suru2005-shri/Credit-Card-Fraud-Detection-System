"""
src/evaluate.py
===============
Model Evaluation Module.

Generates and saves:
  - Classification report (text)
  - Confusion matrix heatmap
  - ROC curve
  - Precision-Recall curve
  - Feature Importance bar chart

All plots are saved to outputs/reports/.

Author  : Fraud Detection System
Version : 1.0
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

# ----------------------------------------------
# Paths
# ----------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
REPORT_DIR  = BASE_DIR / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------
# Plot style
# ----------------------------------------------
plt.rcParams.update({
    "figure.facecolor" : "#0f1117",
    "axes.facecolor"   : "#1a1d27",
    "axes.edgecolor"   : "#3d4263",
    "axes.labelcolor"  : "#c8ccde",
    "xtick.color"      : "#c8ccde",
    "ytick.color"      : "#c8ccde",
    "text.color"       : "#c8ccde",
    "grid.color"       : "#2a2d3a",
    "grid.linestyle"   : "--",
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
})

PALETTE = {
    "fraud"     : "#ff4d6d",
    "legit"     : "#00c9a7",
    "highlight" : "#7b61ff",
    "secondary" : "#ffd166",
}


# ==============================================
# 1. Classification Report (text)
# ==============================================

def print_classification_report(y_true, y_pred, save: bool = True):
    """Print and optionally save the sklearn classification report."""
    report = classification_report(
        y_true, y_pred,
        target_names=["Legitimate (0)", "Fraud (1)"],
        digits=4,
    )
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    if save:
        path = REPORT_DIR / "classification_report.txt"
        path.write_text(report)
        print(f"[Evaluate] Report saved -> {path}")

    return report


# ==============================================
# 2. Confusion Matrix
# ==============================================

def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix"):
    """Plot and save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Legit (0)", "Fraud (1)"]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f1117")

    sns.heatmap(
        cm,
        annot       = True,
        fmt         = "d",
        cmap        = "RdPu",
        linewidths  = 2,
        linecolor   = "#0f1117",
        xticklabels = labels,
        yticklabels = labels,
        ax          = ax,
        annot_kws   = {"size": 16, "weight": "bold"},
    )

    ax.set_title(title, pad=15, fontsize=15, fontweight="bold", color="#ffffff")
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label",      labelpad=10)

    # Annotation for TP / TN / FP / FN
    tn, fp, fn, tp = cm.ravel()
    info = (f"TN={tn:,}  FP={fp:,}\n"
            f"FN={fn:,}  TP={tp:,}\n"
            f"Recall={tp/(tp+fn):.3f}  Precision={tp/(tp+fp):.3f}")
    ax.text(
        1.02, 0.5, info,
        transform       = ax.transAxes,
        va              = "center",
        color           = "#c8ccde",
        fontsize        = 10,
        bbox            = dict(boxstyle="round,pad=0.4",
                               facecolor="#1a1d27", edgecolor="#3d4263"),
    )

    plt.tight_layout()
    out = REPORT_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Confusion matrix saved -> {out}")
    return out


# ==============================================
# 3. ROC Curve
# ==============================================

def plot_roc_curve(y_true, y_proba, model_name: str = "Model"):
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f1117")

    ax.plot(fpr, tpr,
            color     = PALETTE["highlight"],
            lw        = 2.5,
            label     = f"{model_name}  (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1],
            color     = PALETTE["fraud"],
            lw        = 1.5,
            linestyle = "--",
            label     = "Random Classifier (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["highlight"])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve - Fraud Detection", fontweight="bold", color="#ffffff")
    ax.legend(loc="lower right", framealpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = REPORT_DIR / "roc_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] ROC curve saved -> {out}")
    return roc_auc, out


# ==============================================
# 4. Precision-Recall Curve
# ==============================================

def plot_pr_curve(y_true, y_proba, model_name: str = "Model"):
    """Plot and save the Precision-Recall curve (more informative than ROC for fraud)."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_prec = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f1117")

    ax.step(recall, precision,
            where  = "post",
            color  = PALETTE["legit"],
            lw     = 2.5,
            label  = f"{model_name}  (AP = {avg_prec:.4f})")
    ax.fill_between(recall, precision, step="post", alpha=0.12, color=PALETTE["legit"])

    baseline = y_true.mean()
    ax.axhline(baseline, color=PALETTE["fraud"], linestyle="--", lw=1.5,
               label=f"Baseline Precision = {baseline:.4f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Fraud Detection",
                 fontweight="bold", color="#ffffff")
    ax.legend(loc="upper right", framealpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = REPORT_DIR / "pr_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] PR curve saved -> {out}")
    return avg_prec, out


# ==============================================
# 5. Feature Importance
# ==============================================

def plot_feature_importance(model, feature_names, top_n: int = 20):
    """
    Plot feature importances.
    Works for tree-based models (RandomForest, XGBoost).
    Falls back to coefficient magnitude for LinearModels.
    """
    try:
        importances = model.feature_importances_
        imp_type    = "Gini Importance"
    except AttributeError:
        importances = np.abs(model.coef_[0])
        imp_type    = "|Coefficient|"

    idx = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx]
    top_values   = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f1117")

    colors = [PALETTE["fraud"] if "V" in f else PALETTE["highlight"]
              for f in top_features]

    bars = ax.barh(top_features[::-1], top_values[::-1],
                   color=colors[::-1], edgecolor="#0f1117", height=0.7)

    for bar, val in zip(bars, top_values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="left", fontsize=9, color="#c8ccde")

    ax.set_xlabel(imp_type)
    ax.set_title(f"Top {top_n} Feature Importances", fontweight="bold", color="#ffffff")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out = REPORT_DIR / "feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Feature importance saved -> {out}")
    return out


# ==============================================
# 6. Full Evaluation Pipeline
# ==============================================

def full_evaluation(model, X_test, y_test, model_name: str = "Best Model"):
    """
    Run all evaluation plots + classification report.

    Returns
    -------
    dict with roc_auc, avg_precision
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report    = print_classification_report(y_test, y_pred)
    cm_path   = plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {model_name}")
    roc_auc, roc_path = plot_roc_curve(y_test, y_proba, model_name)
    ap, pr_path       = plot_pr_curve(y_test, y_proba, model_name)
    fi_path           = plot_feature_importance(model, list(X_test.columns))

    return {
        "model_name"    : model_name,
        "roc_auc"       : roc_auc,
        "avg_precision" : ap,
        "cm_path"       : str(cm_path),
        "roc_path"      : str(roc_path),
        "pr_path"       : str(pr_path),
        "fi_path"       : str(fi_path),
    }
