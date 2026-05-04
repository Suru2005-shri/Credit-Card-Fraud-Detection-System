"""
src/preprocess.py
=================
Data Cleaning, Feature Engineering & Preprocessing Pipeline.

Steps performed:
  1. Drop duplicates & handle missing values
  2. Feature Engineering  (hour-of-day, log-amount, amount bins)
  3. Scaling             (StandardScaler on Time & Amount)
  4. Train / Validation / Test split  (stratified)
  5. SMOTE oversampling on training set only

Author  : Fraud Detection System
Version : 1.0
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from imblearn.over_sampling  import SMOTE

import joblib

# ----------------------------------------------
# Paths
# ----------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent
SCALER_PATH   = BASE_DIR / "models" / "scaler.pkl"

RANDOM_SEED   = 42


# ==============================================
# 1. Load & basic cleaning
# ==============================================

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load CSV, drop duplicates, confirm no nulls.

    Returns
    -------
    pd.DataFrame  - clean dataframe
    """
    print(f"[Preprocess] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    before = len(df)
    df = df.drop_duplicates()
    after  = len(df)
    print(f"[Preprocess] Dropped {before - after} duplicate rows. Remaining: {after:,}")

    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"[Preprocess] [!] Found {null_count} nulls - filling with column medians.")
        df = df.fillna(df.median(numeric_only=True))
    else:
        print("[Preprocess] No null values found. OK")

    return df


# ==============================================
# 2. Feature Engineering
# ==============================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create informative derived features that help the model
    detect fraud patterns.

    New columns:
      - Hour          : hour of day extracted from Time (seconds)
      - Is_Night      : 1 if hour in {0-5, 23}, else 0
      - Log_Amount    : log1p(Amount) - reduces skewness
      - Amount_Bin    : categorical bucket of transaction size
    """
    df = df.copy()

    # Hour of day
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Hour"] = df["Hour"].astype(int)

    # Night flag (fraud clusters here)
    df["Is_Night"] = df["Hour"].apply(lambda h: 1 if h in [0, 1, 2, 3, 4, 23] else 0)

    # Log-transform amount (heavy right skew)
    df["Log_Amount"] = np.log1p(df["Amount"])

    # Amount bucket
    bins   = [0, 10, 50, 200, 500, np.inf]
    labels = ["micro", "small", "medium", "large", "very_large"]
    df["Amount_Bin"] = pd.cut(df["Amount"], bins=bins, labels=labels).astype(str)

    print("[Preprocess] Feature engineering complete. New columns: "
          "Hour, Is_Night, Log_Amount, Amount_Bin")
    return df


# ==============================================
# 3. Scaling
# ==============================================

def scale_features(
    df: pd.DataFrame,
    fit: bool = True,
    scaler_path: str | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    StandardScale 'Time' and 'Amount' columns (V-features already PCA-scaled).

    Parameters
    ----------
    df          : DataFrame to scale
    fit         : if True, fit a new scaler; if False, load from disk
    scaler_path : path to load/save the scaler

    Returns
    -------
    (scaled_df, scaler)
    """
    path = Path(scaler_path) if scaler_path else SCALER_PATH

    if fit:
        scaler = StandardScaler()
        df[["Time", "Amount", "Log_Amount"]] = scaler.fit_transform(
            df[["Time", "Amount", "Log_Amount"]]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, path)
        print(f"[Preprocess] Scaler fitted and saved -> {path}")
    else:
        scaler = joblib.load(path)
        df[["Time", "Amount", "Log_Amount"]] = scaler.transform(
            df[["Time", "Amount", "Log_Amount"]]
        )
        print(f"[Preprocess] Scaler loaded from {path}")

    return df, scaler


# ==============================================
# 4. Train / Val / Test Split
# ==============================================

def split_data(
    df: pd.DataFrame,
    target: str = "Class",
    test_size: float = 0.20,
    val_size: float  = 0.10,
) -> tuple:
    """
    Stratified 70 / 10 / 20 split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Drop non-numeric engineering columns before modelling
    drop_cols = [target, "Amount_Bin"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        stratify     = y,
        random_state = RANDOM_SEED,
    )

    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size    = relative_val,
        stratify     = y_temp,
        random_state = RANDOM_SEED,
    )

    print(f"[Preprocess] Split -> Train: {len(X_train):,} | "
          f"Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"[Preprocess] Fraud in test set: {y_test.sum()} / {len(y_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==============================================
# 5. SMOTE Oversampling (training only)
# ==============================================

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.1,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to balance training data.

    sampling_strategy = 0.1 -> fraud will be 10 % of majority class
    (avoids too much synthetic data while giving model enough signal).

    Parameters
    ----------
    X_train, y_train         : training features & labels
    sampling_strategy        : desired minority/majority ratio post-SMOTE

    Returns
    -------
    (X_resampled, y_resampled)
    """
    print(f"[Preprocess] SMOTE: before -> {dict(y_train.value_counts())}")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_SEED)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[Preprocess] SMOTE: after  -> {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


# ==============================================
# Full pipeline (used by main.py)
# ==============================================

def run_preprocessing(csv_path: str) -> tuple:
    """
    Convenience wrapper that runs the full preprocessing pipeline.

    Returns
    -------
    X_train_sm, X_val, X_test, y_train_sm, y_val, y_test
    """
    df = load_and_clean(csv_path)
    df = feature_engineering(df)
    df, _  = scale_features(df, fit=True)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    return X_train_sm, X_val, X_test, y_train_sm, y_val, y_test


# ----------------------------------------------
if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "creditcard.csv"
    result = run_preprocessing(str(data_path))
    print("Preprocessing pipeline complete.")
