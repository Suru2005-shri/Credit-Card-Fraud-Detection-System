# -*- coding: utf-8 -*-
"""
src/predict.py
==============
Prediction & Alert Module.

Functions:
  - predict_single()   : classify one transaction dict
  - predict_batch()    : classify a CSV of transactions
  - simulate_live()    : simulate real-time fraud monitoring

Author  : Fraud Detection System
Version : 1.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import time
import random

# --------------------------------------------------------
# Paths
# --------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FEAT_PATH   = BASE_DIR / "models" / "feature_names.pkl"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ========================================================
# Internal helpers
# ========================================================

def _load_artifacts():
    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEAT_PATH) if FEAT_PATH.exists() else None
    return model, scaler, feature_names


def _engineer_row(row: dict) -> dict:
    """Apply same feature engineering as preprocess.py."""
    row = dict(row)
    row["Hour"]       = int((row.get("Time", 0) // 3600) % 24)
    row["Is_Night"]   = 1 if row["Hour"] in [0, 1, 2, 3, 4, 23] else 0
    row["Log_Amount"] = float(np.log1p(row.get("Amount", 0)))
    return row


def _scale_row(row: pd.DataFrame, scaler) -> pd.DataFrame:
    row  = row.copy()
    cols = [c for c in ["Time", "Amount", "Log_Amount"] if c in row.columns]
    row[cols] = scaler.transform(row[cols])
    return row


# ========================================================
# 1. Single Transaction Prediction
# ========================================================

def predict_single(transaction: dict, threshold: float = 0.5) -> dict:
    """
    Predict fraud probability for a single transaction.

    Parameters
    ----------
    transaction : dict with keys matching training features
    threshold   : probability cutoff for fraud label

    Returns
    -------
    dict: { "prediction": "FRAUD"|"LEGIT", "probability": float,
            "risk_level": str, "alert": str }
    """
    model, scaler, feature_names = _load_artifacts()

    engineered = _engineer_row(transaction)
    row_df     = pd.DataFrame([engineered])
    row_df     = row_df.drop(columns=["Amount_Bin"], errors="ignore")
    # Reindex to match training column order
    if feature_names is not None:
        for col in feature_names:
            if col not in row_df.columns:
                row_df[col] = 0.0
        row_df = row_df[feature_names]
    row_scaled = _scale_row(row_df, scaler)

    proba      = model.predict_proba(row_scaled)[0, 1]
    prediction = "FRAUD" if proba >= threshold else "LEGIT"
    risk_level = _risk_level(proba)
    alert      = _compose_alert(transaction, proba, prediction)

    return {
        "prediction" : prediction,
        "probability": round(float(proba), 4),
        "risk_level" : risk_level,
        "alert"      : alert,
    }


def _risk_level(prob: float) -> str:
    if prob >= 0.80:  return "[CRITICAL]"
    if prob >= 0.50:  return "[HIGH]"
    if prob >= 0.20:  return "[MEDIUM]"
    return                   "[LOW]"


def _compose_alert(txn: dict, prob: float, pred: str) -> str:
    if pred == "FRAUD":
        return (
            f"[!] FRAUD ALERT | Amount: ${txn.get('Amount', 0):.2f} | "
            f"Risk: {prob*100:.1f}% | Hour: {int((txn.get('Time', 0)//3600)%24):02d}:xx"
        )
    return f"[OK] Transaction cleared | Risk: {prob*100:.1f}%"


# ========================================================
# 2. Batch Prediction
# ========================================================

def predict_batch(
    df: pd.DataFrame,
    threshold: float = 0.5,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Predict on a DataFrame of transactions.

    Returns
    -------
    df with added columns: Fraud_Prob, Predicted_Class, Risk_Level
    """
    model, scaler, feature_names = _load_artifacts()

    df = df.copy()
    df["Hour"]       = (df["Time"] // 3600).astype(int) % 24
    df["Is_Night"]   = df["Hour"].apply(lambda h: 1 if h in [0,1,2,3,4,23] else 0)
    df["Log_Amount"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Amount_Bin", "Class"], errors="ignore")
    df[["Time", "Amount", "Log_Amount"]] = scaler.transform(
        df[["Time", "Amount", "Log_Amount"]]
    )

    proba = model.predict_proba(df)[:, 1]
    df["Fraud_Prob"]      = proba.round(4)
    df["Predicted_Class"] = (proba >= threshold).astype(int)
    df["Risk_Level"]      = [_risk_level(p) for p in proba]

    n_fraud = df["Predicted_Class"].sum()
    print(f"[Predict] Batch: {n_fraud}/{len(df)} flagged "
          f"({n_fraud/len(df)*100:.2f}%)")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[Predict] Predictions saved -> {save_path}")

    return df


# ========================================================
# 3. Live Simulation
# ========================================================

def simulate_live(n_transactions: int = 20, delay: float = 0.4):
    """
    Simulate a live transaction stream and print alerts in real time.
    Randomly injects fraud transactions for demonstration.
    """
    model, scaler, _ = _load_artifacts()
    np.random.seed(RANDOM_SEED)

    print("\n" + "=" * 70)
    print("  [*] LIVE FRAUD DETECTION SIMULATION STARTED")
    print("=" * 70)

    flagged = 0
    for i in range(1, n_transactions + 1):
        is_fraud_sim = (random.random() < 0.15)   # 15% injected fraud rate

        if is_fraud_sim:
            amount = round(random.uniform(500, 4000), 2)
            hour   = random.choice([0, 1, 2, 3, 23])
        else:
            amount = round(np.random.lognormal(3.5, 1.2), 2)
            hour   = random.randint(8, 21)

        txn = {
            "Time"  : hour * 3600 + random.randint(0, 3599),
            "Amount": amount,
            **{f"V{j}": round(float(np.random.normal(0, 1.5 if is_fraud_sim else 1.0)), 4)
               for j in range(1, 29)},
        }
        if is_fraud_sim:
            txn["V3"]  = txn["V3"]  - 3.5
            txn["V14"] = txn["V14"] - 7.0

        result = predict_single(txn, threshold=0.35)

        status_icon = "[FRAUD]" if result["prediction"] == "FRAUD" else "[OK]"
        if result["prediction"] == "FRAUD":
            flagged += 1

        print(
            f"  [{i:03d}] {status_icon:<8} "
            f"${amount:>8.2f}  @{hour:02d}:xx  |  "
            f"Risk: {result['probability']*100:5.1f}%  |  "
            f"{result['risk_level']}"
        )
        time.sleep(delay)

    print("=" * 70)
    print(f"  Simulation complete: {flagged}/{n_transactions} transactions flagged.")
    print("=" * 70 + "\n")


# --------------------------------------------------------
if __name__ == "__main__":
    simulate_live(n_transactions=15, delay=0.2)
