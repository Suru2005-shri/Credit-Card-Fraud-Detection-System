# -*- coding: utf-8 -*-
"""
main.py
=======
Credit Card Fraud Detection System - Master Pipeline.

Usage:
    python main.py                  # full pipeline
    python main.py --simulate       # live simulation only
    python main.py --eda            # EDA only

Workflow:
  1. Generate synthetic dataset
  2. Preprocess & engineer features
  3. Train & select best model
  4. Evaluate on test set
  5. Generate EDA + evaluation visualizations
  6. Run live simulation
  7. Predict a sample transaction

Author  : Fraud Detection System
Version : 1.0
"""

import sys
import time
import argparse
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_generator import generate_dataset
from src.preprocess     import run_preprocessing, load_and_clean, feature_engineering
from src.train          import train_all_models, save_best_model, load_model
from src.evaluate       import full_evaluation
from src.visualize      import run_full_eda
from src.predict        import predict_single, simulate_live

# --------------------------------------------------------
# Paths
# --------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"


# ========================================================
# Banner
# ========================================================
def print_banner():
    print("""
+--------------------------------------------------------------+
|   [CC]  CREDIT CARD FRAUD DETECTION SYSTEM  v1.0            |
|   Industry-Grade ML Pipeline  |  Python 3.11                |
+--------------------------------------------------------------+
""")


# ========================================================
# Full Pipeline
# ========================================================
def run_full_pipeline():
    t_start = time.time()
    print_banner()

    # -------------------------------------------------
    # STEP 1: Data Generation
    # -------------------------------------------------
    print("\n" + "-" * 60)
    print("  STEP 1/7: Generating Synthetic Dataset")
    print("-" * 60)
    if DATA_PATH.exists():
        print(f"  [Skip] Dataset already exists -> {DATA_PATH}")
    else:
        generate_dataset(save_path=str(DATA_PATH))

    # -------------------------------------------------
    # STEP 2: EDA (before preprocessing)
    # -------------------------------------------------
    print("\n" + "-" * 60)
    print("  STEP 2/7: Exploratory Data Analysis")
    print("-" * 60)
    raw_df = load_and_clean(str(DATA_PATH))
    raw_df = feature_engineering(raw_df)
    print(f"\n  Dataset shape      : {raw_df.shape}")
    print(f"  Fraud transactions : {raw_df['Class'].sum():,} "
          f"({raw_df['Class'].mean()*100:.3f}%)")
    print(f"  Amount range       : ${raw_df['Amount'].min():.2f} "
          f"- ${raw_df['Amount'].max():.2f}")
    print(f"  Columns            : {list(raw_df.columns[:8])} ...")
    print(f"\n  Sample data:\n{raw_df[['Time','Amount','Hour','Is_Night','Class']].head()}")

    run_full_eda(raw_df)

    # -------------------------------------------------
    # STEP 3: Preprocessing
    # -------------------------------------------------
    print("\n" + "-" * 60)
    print("  STEP 3/7: Preprocessing & Feature Engineering")
    print("-" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing(str(DATA_PATH))

    # -------------------------------------------------
    # STEP 4: Model Training
    # -------------------------------------------------
    print("\n" + "-" * 60)
    print("  STEP 4/7: Training Models")
    print("-" * 60)
    results, best_name = train_all_models(X_train, y_train, X_val, y_val)
    save_best_model(results, best_name, feature_names=list(X_train.columns))

    # -------------------------------------------------
    # STEP 5: Evaluation
    # -------------------------------------------------
    print("\n" + "-" * 60)
    print("  STEP 5/7: Evaluating Best Model on Test Set")
    print("-" * 60)
    best_model   = load_model()
    eval_metrics = full_evaluation(best_model, X_test, y_test, model_name=best_name)

    from sklearn.metrics import recall_score, precision_score, f1_score
    y_pred = best_model.predict(X_test)
    metrics_for_dash = {
        "roc_auc"       : eval_metrics["roc_auc"],
        "avg_precision" : eval_metrics["avg_precision"],
        "recall"        : recall_score(y_test, y_pred, zero_division=0),
        "precision"     : precision_score(y_test, y_pred, zero_division=0),
        "f1"            : f1_score(y_test, y_pred, zero_division=0),
        "model_name"    : best_name,
    }

    from src.visualize import plot_summary_dashboard
    plot_summary_dashboard(raw_df, metrics_for_dash)

    # -------------------------------------------------
    # STEP 6: Live Simulation
    # -------------------------------------------------
    print("\n" + "-" * 60)
    print("  STEP 6/7: Live Fraud Simulation")
    print("-" * 60)
    simulate_live(n_transactions=15, delay=0.15)

    # -------------------------------------------------
    # STEP 7: Sample Prediction
    # -------------------------------------------------
    print("\n" + "-" * 60)
    print("  STEP 7/7: Sample Transaction Predictions")
    print("-" * 60)

    import numpy as np
    sample_transactions = [
        {
            "Time": 36000, "Amount": 45.50,
            **{f"V{i}": round(float(np.random.normal(0, 0.8)), 4) for i in range(1, 29)},
            "desc": "Grocery store - $45.50 @ 10 AM",
        },
        {
            "Time": 7200, "Amount": 3200.00,
            **{f"V{i}": round(float(np.random.normal(-2 if i in [3, 14] else 0, 1.5)), 4)
               for i in range(1, 29)},
            "desc": "Online purchase - $3200 @ 2 AM",
        },
        {
            "Time": 3600, "Amount": 0.99,
            **{f"V{i}": round(float(np.random.normal(-1.5 if i in [10, 12] else 0, 1.2)), 4)
               for i in range(1, 29)},
            "desc": "Micro-transaction probe - $0.99 @ 1 AM",
        },
    ]

    for txn in sample_transactions:
        desc = txn.pop("desc")
        result = predict_single(txn)
        print(f"\n  Transaction: {desc}")
        print(f"  -> Prediction : {result['prediction']}")
        print(f"  -> Risk Level : {result['risk_level']}")
        print(f"  -> Probability: {result['probability']*100:.2f}%")
        print(f"  -> {result['alert']}")

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 62)
    print(f"  [DONE]  PIPELINE COMPLETE  |  Time: {elapsed:.1f}s")
    print(f"  Best Model  : {best_name}")
    print(f"  ROC-AUC     : {eval_metrics['roc_auc']:.4f}")
    print(f"  Avg Prec.   : {eval_metrics['avg_precision']:.4f}")
    print(f"  Charts saved: images/")
    print(f"  Reports     : outputs/reports/")
    print(f"  Model saved : models/best_model.pkl")
    print("=" * 62 + "\n")


# ========================================================
# EDA-only mode
# ========================================================
def run_eda_only():
    print_banner()
    print("\n[Mode] EDA Only")
    df = load_and_clean(str(DATA_PATH))
    df = feature_engineering(df)
    run_full_eda(df)
    print("\n[OK] EDA complete. Charts saved to images/")


# ========================================================
# Simulation-only mode
# ========================================================
def run_simulation_only():
    print_banner()
    print("\n[Mode] Live Simulation")
    simulate_live(n_transactions=25, delay=0.3)


# ========================================================
# CLI Entry Point
# ========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection System"
    )
    parser.add_argument("--simulate", action="store_true",
                        help="Run live transaction simulation only")
    parser.add_argument("--eda",      action="store_true",
                        help="Run EDA visualizations only")
    args = parser.parse_args()

    if args.simulate:
        run_simulation_only()
    elif args.eda:
        run_eda_only()
    else:
        run_full_pipeline()
