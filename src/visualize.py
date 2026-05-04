"""
src/visualize.py
================
EDA & Visualization Module.

Generates:
  1. Class distribution bar chart
  2. Transaction amount distribution (legit vs fraud)
  3. Hour-of-day fraud heatmap
  4. Correlation heatmap (top V-features)
  5. Fraud probability distribution
  6. Summary dashboard (composite figure)

All outputs saved to outputs/reports/ and images/.

Author  : Fraud Detection System
Version : 1.0
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ----------------------------------------------
# Paths
# ----------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
REPORT_DIR  = BASE_DIR / "outputs" / "reports"
IMAGE_DIR   = BASE_DIR / "images"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------
# Style
# ----------------------------------------------
DARK_BG  = "#0f1117"
CARD_BG  = "#1a1d27"
BORDER   = "#3d4263"
TEXT     = "#c8ccde"
WHITE    = "#ffffff"
FRAUD_C  = "#ff4d6d"
LEGIT_C  = "#00c9a7"
ACCENT   = "#7b61ff"
GOLD     = "#ffd166"

plt.rcParams.update({
    "figure.facecolor" : DARK_BG,
    "axes.facecolor"   : CARD_BG,
    "axes.edgecolor"   : BORDER,
    "axes.labelcolor"  : TEXT,
    "xtick.color"      : TEXT,
    "ytick.color"      : TEXT,
    "text.color"       : TEXT,
    "grid.color"       : BORDER,
    "grid.linestyle"   : "--",
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
})


def _save(fig, filename: str) -> Path:
    out = IMAGE_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Visualize] Saved -> {out}")
    return out


# ==============================================
# 1. Class Distribution
# ==============================================

def plot_class_distribution(df: pd.DataFrame):
    counts = df["Class"].value_counts()
    labels = ["Legitimate", "Fraud"]
    colors = [LEGIT_C, FRAUD_C]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Transaction Class Distribution", fontsize=16,
                 fontweight="bold", color=WHITE, y=1.02)

    # Bar chart
    axes[0].bar(labels, [counts[0], counts[1]], color=colors,
                edgecolor=DARK_BG, width=0.5)
    for i, (v, l) in enumerate(zip([counts[0], counts[1]], labels)):
        axes[0].text(i, v + counts[0] * 0.01, f"{v:,}", ha="center",
                     fontsize=12, color=WHITE, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Transaction Counts")
    axes[0].grid(axis="y", alpha=0.3)

    # Pie chart
    wedge_props = dict(width=0.55, edgecolor=DARK_BG, linewidth=2)
    axes[1].pie(
        [counts[0], counts[1]],
        labels      = [f"Legitimate\n{counts[0]:,}", f"Fraud\n{counts[1]:,}"],
        colors      = colors,
        autopct     = "%1.2f%%",
        startangle  = 90,
        wedgeprops  = wedge_props,
        textprops   = dict(color=WHITE, fontsize=11),
        pctdistance = 0.75,
    )
    axes[1].set_title("Class Proportion")

    plt.tight_layout()
    return _save(fig, "01_class_distribution.png")


# ==============================================
# 2. Amount Distribution
# ==============================================

def plot_amount_distribution(df: pd.DataFrame):
    legit = df[df["Class"] == 0]["Amount"]
    fraud = df[df["Class"] == 1]["Amount"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transaction Amount Distribution", fontsize=16,
                 fontweight="bold", color=WHITE, y=1.02)

    # KDE on log scale
    axes[0].hist(np.log1p(legit), bins=60, alpha=0.7, color=LEGIT_C,
                 label="Legitimate", density=True, edgecolor=DARK_BG)
    axes[0].hist(np.log1p(fraud), bins=30, alpha=0.8, color=FRAUD_C,
                 label="Fraud", density=True, edgecolor=DARK_BG)
    axes[0].set_xlabel("log(Amount + 1)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Amount Distribution (log scale)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Box plots
    data_to_plot = [np.log1p(legit), np.log1p(fraud)]
    bp = axes[1].boxplot(data_to_plot, patch_artist=True,
                         medianprops=dict(color=GOLD, lw=2),
                         whiskerprops=dict(color=TEXT),
                         capprops=dict(color=TEXT),
                         flierprops=dict(marker="o", color=FRAUD_C, alpha=0.4, ms=3))
    bp["boxes"][0].set_facecolor(LEGIT_C + "66")
    bp["boxes"][1].set_facecolor(FRAUD_C + "66")
    axes[1].set_xticklabels(["Legitimate", "Fraud"])
    axes[1].set_ylabel("log(Amount + 1)")
    axes[1].set_title("Amount Box Plot")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return _save(fig, "02_amount_distribution.png")


# ==============================================
# 3. Hour-of-day Fraud Heatmap
# ==============================================

def plot_hour_fraud_heatmap(df: pd.DataFrame):
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600).astype(int) % 24

    hour_fraud = (
        df.groupby("Hour")["Class"]
          .agg(["sum", "count"])
          .rename(columns={"sum": "Fraud", "count": "Total"})
    )
    hour_fraud["Fraud_Rate"] = hour_fraud["Fraud"] / hour_fraud["Total"]
    hour_fraud = hour_fraud.reindex(range(24), fill_value=0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Fraud Activity by Hour of Day", fontsize=16,
                 fontweight="bold", color=WHITE)

    # Fraud count per hour
    axes[0].bar(hour_fraud.index, hour_fraud["Fraud"],
                color=[FRAUD_C if h in [0,1,2,3,4,23] else ACCENT
                       for h in hour_fraud.index],
                edgecolor=DARK_BG)
    axes[0].set_ylabel("Fraud Count")
    axes[0].set_title("Fraud Transactions per Hour")
    axes[0].set_xticks(range(24))
    axes[0].grid(axis="y", alpha=0.3)

    # Fraud rate heatmap
    rate_matrix = hour_fraud["Fraud_Rate"].values.reshape(1, -1)
    sns.heatmap(
        rate_matrix,
        ax          = axes[1],
        cmap        = "RdPu",
        linewidths  = 0.5,
        linecolor   = DARK_BG,
        xticklabels = [f"{h:02d}" for h in range(24)],
        yticklabels = ["Fraud\nRate"],
        annot       = True,
        fmt         = ".3f",
        annot_kws   = {"size": 7},
    )
    axes[1].set_title("Fraud Rate Heatmap")
    axes[1].set_xlabel("Hour of Day (24h)")

    plt.tight_layout()
    return _save(fig, "03_hour_fraud_heatmap.png")


# ==============================================
# 4. Correlation Heatmap (V-features)
# ==============================================

def plot_correlation_heatmap(df: pd.DataFrame):
    # Top V-features most correlated with Class
    v_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    corr_with_class = df[v_cols].corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    top_features    = corr_with_class.head(12).index.tolist() + ["Class"]

    corr_matrix = df[top_features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask    = np.zeros_like(corr_matrix, dtype=bool)
    np.fill_diagonal(mask, True)                   # hide diagonal

    sns.heatmap(
        corr_matrix,
        ax          = ax,
        mask        = mask,
        cmap        = "coolwarm",
        center      = 0,
        annot       = True,
        fmt         = ".2f",
        linewidths  = 0.5,
        linecolor   = DARK_BG,
        annot_kws   = {"size": 8},
        square      = True,
    )
    ax.set_title("Feature Correlation Heatmap (Top 12 + Class)",
                 fontweight="bold", color=WHITE, pad=15)

    plt.tight_layout()
    return _save(fig, "04_correlation_heatmap.png")


# ==============================================
# 5. Summary Dashboard
# ==============================================

def plot_summary_dashboard(df: pd.DataFrame, metrics: dict | None = None):
    """
    4-panel summary figure combining key EDA insights.
    metrics = { "roc_auc": float, "avg_precision": float, ... }
    """
    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle("Credit Card Fraud Detection - Summary Dashboard",
                 fontsize=18, fontweight="bold", color=WHITE, y=1.01)

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # -- Panel A: Class distribution ----------
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["Class"].value_counts()
    ax1.bar(["Legitimate", "Fraud"], [counts[0], counts[1]],
            color=[LEGIT_C, FRAUD_C], edgecolor=DARK_BG, width=0.5)
    ax1.set_title("A. Class Distribution")
    ax1.set_ylabel("Count")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate([counts[0], counts[1]]):
        ax1.text(i, v, f"{v:,}", ha="center", va="bottom",
                 fontsize=10, color=WHITE, fontweight="bold")

    # -- Panel B: Amount distribution ---------
    ax2 = fig.add_subplot(gs[0, 1])
    legit = np.log1p(df[df["Class"] == 0]["Amount"])
    fraud = np.log1p(df[df["Class"] == 1]["Amount"])
    ax2.hist(legit, bins=50, alpha=0.7, color=LEGIT_C,
             label="Legitimate", density=True)
    ax2.hist(fraud, bins=25, alpha=0.85, color=FRAUD_C,
             label="Fraud", density=True)
    ax2.set_title("B. Amount Distribution (log)")
    ax2.set_xlabel("log(Amount + 1)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # -- Panel C: Hour fraud rate --------------
    ax3 = fig.add_subplot(gs[1, 0])
    df2 = df.copy()
    df2["Hour"] = (df2["Time"] // 3600).astype(int) % 24
    hour_rate = df2.groupby("Hour")["Class"].mean().reindex(range(24), fill_value=0)
    ax3.bar(hour_rate.index, hour_rate.values * 100,
            color=[FRAUD_C if h in [0,1,2,3,4,23] else ACCENT for h in hour_rate.index])
    ax3.set_title("C. Fraud Rate by Hour (%)")
    ax3.set_xlabel("Hour (24h)")
    ax3.set_ylabel("Fraud Rate (%)")
    ax3.set_xticks(range(0, 24, 2))
    ax3.grid(axis="y", alpha=0.3)

    # -- Panel D: Model metrics ----------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_axis_off()
    if metrics:
        metric_text = (
            f"Model Performance\n"
            f"{'-'*30}\n"
            f"ROC-AUC          : {metrics.get('roc_auc', 0):.4f}\n"
            f"Avg Precision    : {metrics.get('avg_precision', 0):.4f}\n"
            f"Recall (Fraud)   : {metrics.get('recall', 0):.4f}\n"
            f"Precision (Fraud): {metrics.get('precision', 0):.4f}\n"
            f"F1 Score         : {metrics.get('f1', 0):.4f}\n"
            f"{'-'*30}\n"
            f"Best Model       : {metrics.get('model_name', 'N/A')}"
        )
    else:
        metric_text = "Run full pipeline\nto see metrics here."
    ax4.text(
        0.5, 0.5, metric_text,
        ha        = "center", va = "center",
        fontsize  = 12,
        transform = ax4.transAxes,
        color     = WHITE,
        fontfamily= "monospace",
        bbox      = dict(boxstyle="round,pad=0.8",
                         facecolor=CARD_BG, edgecolor=ACCENT, lw=1.5),
    )
    ax4.set_title("D. Model Performance")

    return _save(fig, "05_summary_dashboard.png")


# ==============================================
# Full EDA runner
# ==============================================

def run_full_eda(df: pd.DataFrame, metrics: dict | None = None):
    """Run all EDA plots."""
    print("\n[Visualize] Generating EDA visualizations ...")
    paths = []
    paths.append(plot_class_distribution(df))
    paths.append(plot_amount_distribution(df))
    paths.append(plot_hour_fraud_heatmap(df))
    paths.append(plot_correlation_heatmap(df))
    paths.append(plot_summary_dashboard(df, metrics))
    print(f"[Visualize] All {len(paths)} charts saved.")
    return paths


# ----------------------------------------------
if __name__ == "__main__":
    from data_generator import generate_dataset
    df = generate_dataset()
    run_full_eda(df)
