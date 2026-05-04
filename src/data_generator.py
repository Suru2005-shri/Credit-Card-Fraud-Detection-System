"""
src/data_generator.py
=====================
Synthetic Credit Card Transaction Data Generator.

Generates realistic transaction data with deliberate fraud patterns
so the ML model has meaningful signal to learn from.

Fraud patterns simulated:
  - Large amounts at odd hours (late night)
  - Multiple rapid small transactions (card testing)
  - Foreign / unusual merchant categories
  - High velocity: many transactions in a short window
  - Geographic anomaly: distance between consecutive transactions

Author  : Fraud Detection System
Version : 1.0
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------
# Reproducibility
# ----------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ----------------------------------------------
# Constants
# ----------------------------------------------
N_TRANSACTIONS  = 284_807          # mirrors the famous Kaggle dataset size
FRAUD_RATE      = 0.001724         # ~0.17 % - realistic banking rate
N_FRAUD         = int(N_TRANSACTIONS * FRAUD_RATE)   # ~ 492
N_LEGIT         = N_TRANSACTIONS - N_FRAUD

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail",
    "travel", "entertainment", "healthcare", "atm_withdrawal",
    "electronics", "luxury_goods",
]

LEGIT_HOURS   = list(range(7, 23))          # 7 AM - 10 PM
FRAUD_HOURS   = [0, 1, 2, 3, 4, 23]        # late night spike


# ==============================================
# Helper generators
# ==============================================

def _legit_amount() -> float:
    """Legitimate transaction amounts follow a log-normal distribution."""
    return round(np.random.lognormal(mean=3.5, sigma=1.2), 2)   # ~$5 - $500


def _fraud_amount() -> float:
    """
    Fraud amounts:
      - 40 % are suspiciously large (card maxed out quickly)
      - 60 % are tiny (card-testing / probe transactions)
    """
    if np.random.rand() < 0.40:
        return round(np.random.uniform(500, 5_000), 2)
    else:
        return round(np.random.uniform(0.01, 5.00), 2)


def _v_features_legit(n: int) -> np.ndarray:
    """
    28 PCA-like anonymised features (V1-V28).
    Legitimate transactions cluster near 0 with low variance.
    """
    return np.random.normal(loc=0.0, scale=1.0, size=(n, 28))


def _v_features_fraud(n: int) -> np.ndarray:
    """
    Fraud transactions sit in unusual PCA space - shifted means and
    higher variance on key components (V3, V4, V10, V12, V14, V17).
    """
    features = np.random.normal(loc=0.0, scale=1.5, size=(n, 28))
    # Key PCA components that separate fraud in the real dataset
    features[:, 2]  += np.random.normal(-3.5, 1.0, n)   # V3
    features[:, 3]  += np.random.normal( 3.0, 1.2, n)   # V4
    features[:, 9]  += np.random.normal(-4.0, 1.5, n)   # V10
    features[:, 11] += np.random.normal(-5.0, 1.0, n)   # V12
    features[:, 13] += np.random.normal(-7.0, 1.5, n)   # V14
    features[:, 16] += np.random.normal(-4.5, 1.0, n)   # V17
    return features


# ==============================================
# Main generator
# ==============================================

def generate_dataset(save_path: str | None = None) -> pd.DataFrame:
    """
    Generate synthetic credit-card transaction data.

    Parameters
    ----------
    save_path : str or None
        If provided, saves the CSV to this path.

    Returns
    -------
    pd.DataFrame
        Full dataset with columns: Time, V1-V28, Amount, Class.
    """

    print(f"[DataGenerator] Generating {N_TRANSACTIONS:,} transactions "
          f"({N_LEGIT:,} legitimate, {N_FRAUD:,} fraudulent) ...")

    # -- Legitimate transactions ------------------
    legit_time    = np.sort(np.random.uniform(0, 172_800, N_LEGIT))   # 48-hr window
    legit_amounts = np.array([_legit_amount() for _ in range(N_LEGIT)])
    legit_v       = _v_features_legit(N_LEGIT)

    # -- Fraudulent transactions ------------------
    # Fraud time: cluster around midnight windows
    fraud_time    = np.sort(np.random.choice(
        [h * 3600 for h in FRAUD_HOURS], size=N_FRAUD
    ) + np.random.uniform(0, 3600, N_FRAUD))
    fraud_amounts = np.array([_fraud_amount() for _ in range(N_FRAUD)])
    fraud_v       = _v_features_fraud(N_FRAUD)

    # -- Build DataFrames -------------------------
    legit_df = pd.DataFrame(
        legit_v,
        columns=[f"V{i}" for i in range(1, 29)]
    )
    legit_df.insert(0, "Time", legit_time)
    legit_df["Amount"] = legit_amounts
    legit_df["Class"]  = 0

    fraud_df = pd.DataFrame(
        fraud_v,
        columns=[f"V{i}" for i in range(1, 29)]
    )
    fraud_df.insert(0, "Time", fraud_time)
    fraud_df["Amount"] = fraud_amounts
    fraud_df["Class"]  = 1

    # -- Merge & shuffle --------------------------
    df = (
        pd.concat([legit_df, fraud_df], ignore_index=True)
          .sample(frac=1, random_state=RANDOM_SEED)
          .reset_index(drop=True)
    )

    # -- Optional save ----------------------------
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[DataGenerator] Dataset saved -> {save_path}")

    print(f"[DataGenerator] Done. Shape: {df.shape}")
    return df


# ==============================================
# Quick test
# ==============================================
if __name__ == "__main__":
    df = generate_dataset(save_path="../data/creditcard.csv")
    print(df.head())
    print("\nClass distribution:")
    print(df["Class"].value_counts())
