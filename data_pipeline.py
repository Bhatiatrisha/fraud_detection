"""
data_pipeline.py
----------------
Generates a realistic 1M-row synthetic transaction dataset and
provides a full preprocessing pipeline (feature engineering,
train/val/test split, SMOTE oversampling).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


# ── 1. SYNTHETIC DATA GENERATION ─────────────────────────────────────────────

def generate_transactions(n: int = 1_000_000, seed: int = 42) -> pd.DataFrame:
    """
    Simulate realistic credit-card transactions.

    Fraud rate ≈ 0.3%  (severe class imbalance, mirrors real-world data)
    """
    rng = np.random.default_rng(seed)

    # --- core fields ---
    transaction_id = np.arange(n)
    amount         = rng.lognormal(mean=4.5, sigma=1.8, size=n).clip(0.5, 25_000)
    hour           = rng.integers(0, 24, size=n)
    day_of_week    = rng.integers(0, 7, size=n)
    merchant_category = rng.choice(
        ["retail", "grocery", "travel", "dining", "entertainment",
         "gas", "online", "atm", "healthcare", "utilities"],
        size=n,
        p=[0.20, 0.18, 0.10, 0.12, 0.08, 0.07, 0.12, 0.05, 0.05, 0.03],
    )
    country = rng.choice(
        ["domestic", "foreign"],
        size=n,
        p=[0.85, 0.15],
    )
    card_type = rng.choice(["credit", "debit", "prepaid"], size=n, p=[0.55, 0.38, 0.07])

    # --- derived / engineered signals ---
    velocity_1h  = rng.integers(1, 20, size=n)   # txns in last 1 h by same card
    velocity_24h = velocity_1h + rng.integers(0, 50, size=n)
    days_since_last_txn = rng.integers(0, 30, size=n)
    card_age_days       = rng.integers(1, 3650, size=n)

    # --- fraud labels with realistic signal injection ---
    fraud_score = (
          0.10 * (amount > 1_500).astype(float)           # high amount
        + 0.12 * (merchant_category == "atm").astype(float)
        + 0.10 * (country == "foreign").astype(float)
        + 0.08 * (hour < 4).astype(float)                  # unusual hour
        + 0.09 * (velocity_1h > 10).astype(float)
        + 0.07 * (card_type == "prepaid").astype(float)
        + 0.06 * (card_age_days < 30).astype(float)        # new card
        + rng.uniform(0, 0.35, size=n)                     # noise
    )
    is_fraud = (fraud_score > 0.55).astype(int)

    # force ≈ 0.3 % fraud rate (realistic)
    fraud_idx    = np.where(is_fraud == 1)[0]
    non_fraud_idx = np.where(is_fraud == 0)[0]
    target_fraud  = int(n * 0.003)
    if len(fraud_idx) > target_fraud:
        drop = rng.choice(fraud_idx, size=len(fraud_idx) - target_fraud, replace=False)
        is_fraud[drop] = 0

    df = pd.DataFrame({
        "transaction_id":      transaction_id,
        "amount":              amount.round(2),
        "hour":                hour,
        "day_of_week":         day_of_week,
        "merchant_category":   merchant_category,
        "country":             country,
        "card_type":           card_type,
        "velocity_1h":         velocity_1h,
        "velocity_24h":        velocity_24h,
        "days_since_last_txn": days_since_last_txn,
        "card_age_days":       card_age_days,
        "is_fraud":            is_fraud,
    })
    return df


# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # log-transform skewed amount
    df["log_amount"]   = np.log1p(df["amount"])
    df["amount_sq"]    = df["amount"] ** 0.5

    # time features
    df["is_night"]     = (df["hour"].between(0, 5)).astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)

    # velocity ratios
    df["velocity_ratio"] = df["velocity_1h"] / (df["velocity_24h"] + 1)

    # card risk proxy
    df["new_card"]     = (df["card_age_days"] < 30).astype(int)

    # encode categoricals
    for col in ["merchant_category", "country", "card_type"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])

    return df


# ── 3. SPLIT & SMOTE ─────────────────────────────────────────────────────────

FEATURE_COLS = [
    "log_amount", "amount_sq", "hour_sin", "hour_cos",
    "is_night", "is_weekend", "velocity_1h", "velocity_24h",
    "velocity_ratio", "days_since_last_txn", "card_age_days",
    "new_card", "merchant_category_enc", "country_enc", "card_type_enc",
]

def prepare_splits(df: pd.DataFrame, smote: bool = True, seed: int = 42):
    """
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    Training set is SMOTE-oversampled when smote=True.
    """
    df = engineer_features(df)
    X, y = df[FEATURE_COLS], df["is_fraud"]

    # 70 / 15 / 15  split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )

    scaler = StandardScaler()
    X_tr   = pd.DataFrame(scaler.fit_transform(X_tr),   columns=FEATURE_COLS)
    X_val  = pd.DataFrame(scaler.transform(X_val),      columns=FEATURE_COLS)
    X_test = pd.DataFrame(scaler.transform(X_test),     columns=FEATURE_COLS)

    if smote:
        sm = SMOTE(sampling_strategy=0.1, random_state=seed)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

    print(f"Train : {X_tr.shape[0]:>8,}  (fraud={y_tr.sum():,})")
    print(f"Val   : {X_val.shape[0]:>8,}  (fraud={y_val.sum():,})")
    print(f"Test  : {X_test.shape[0]:>8,}  (fraud={y_test.sum():,})")

    return X_tr, X_val, X_test, y_tr, y_val, y_test, scaler
