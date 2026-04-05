"""
risk_scorer.py
--------------
Production-ready risk scoring wrapper around the trained XGBoost model.
Maps raw probability → risk tier → recommended action.
"""

import numpy as np
import pandas as pd
from src.data_pipeline import engineer_features, FEATURE_COLS


RISK_TIERS = [
    (0.80, "CRITICAL",  "Block & alert customer immediately"),
    (0.50, "HIGH",      "Flag for manual review"),
    (0.25, "MEDIUM",    "Step-up authentication"),
    (0.00, "LOW",       "Approve — monitor passively"),
]


class FraudRiskScorer:
    """
    Wraps a trained XGBoost model and StandardScaler to score
    individual or batch transactions.

    Parameters
    ----------
    model   : trained XGBClassifier
    scaler  : fitted StandardScaler
    """

    def __init__(self, model, scaler):
        self.model  = model
        self.scaler = scaler

    def score(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Score a DataFrame of raw transactions.

        Returns the original DataFrame plus:
          risk_score  : float  [0, 1]
          risk_tier   : str    LOW / MEDIUM / HIGH / CRITICAL
          action      : str    recommended action
        """
        df = engineer_features(transactions)
        X  = df[FEATURE_COLS]
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)[:, 1]

        tiers   = []
        actions = []
        for p in proba:
            for threshold, tier, action in RISK_TIERS:
                if p >= threshold:
                    tiers.append(tier)
                    actions.append(action)
                    break

        out = transactions.copy()
        out["risk_score"] = proba.round(4)
        out["risk_tier"]  = tiers
        out["action"]     = actions
        return out

    def score_record(self, **kwargs) -> dict:
        """Score a single transaction passed as keyword args."""
        df  = pd.DataFrame([kwargs])
        out = self.score(df)
        return out.iloc[0].to_dict()


# ── DEMO ─────────────────────────────────────────────────────────────────────

def demo_scoring(scorer, n=8, seed=99):
    """Generate n random transactions and show scored output."""
    rng = np.random.default_rng(seed)
    demo_txns = pd.DataFrame({
        "transaction_id":      np.arange(n),
        "amount":              rng.choice([15.0, 380.5, 1200.0, 8500.0, 50.0, 2200.0, 99.9, 4500.0]),
        "hour":                rng.integers(0, 24, n),
        "day_of_week":         rng.integers(0, 7, n),
        "merchant_category":   rng.choice(["retail","atm","online","grocery"], n),
        "country":             rng.choice(["domestic","foreign"], n, p=[0.7,0.3]),
        "card_type":           rng.choice(["credit","debit","prepaid"], n, p=[0.5,0.4,0.1]),
        "velocity_1h":         rng.integers(1, 15, n),
        "velocity_24h":        rng.integers(1, 40, n),
        "days_since_last_txn": rng.integers(0, 20, n),
        "card_age_days":       rng.integers(5, 2000, n),
    })
    scored = scorer.score(demo_txns)
    cols   = ["transaction_id","amount","risk_score","risk_tier","action"]
    return scored[cols]
