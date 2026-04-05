"""
models.py
---------
Trains and benchmarks Random Forest, XGBoost, and LightGBM.
Primary metric: PR-AUC (better than ROC-AUC for imbalanced data).
"""

import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, classification_report,
    confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


# ── HYPERPARAMETERS (tuned for fraud detection) ───────────────────────────────

LR_PARAMS = dict(
    C=0.1, class_weight="balanced", max_iter=1000, random_state=42
)

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=30,      # handles imbalance
    eval_metric="aucpr",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

LGB_PARAMS = dict(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    is_unbalance=True,
    metric="average_precision",
    early_stopping_round=30,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


# ── TRAINING ─────────────────────────────────────────────────────────────────

def train_all(X_tr, X_val, y_tr, y_val):
    """Train all four models and return them in a dict."""
    models = {}

    print("\n[1/4] Logistic Regression (baseline)…")
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_tr, y_tr)
    models["LogisticRegression"] = lr

    print("[2/4] Random Forest…")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_tr, y_tr)
    models["RandomForest"] = rf

    print("[3/4] XGBoost…")
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    models["XGBoost"] = xgb_model

    print("[4/4] LightGBM…")
    lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    models["LightGBM"] = lgb_model

    print("\n✓ All models trained.\n")
    return models


# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate(models: dict, X_test, y_test) -> pd.DataFrame:
    """Compute PR-AUC, ROC-AUC, precision, recall, F1 for each model."""
    rows = []
    threshold = 0.5

    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        pred  = (proba >= threshold).astype(int)

        pr_auc  = average_precision_score(y_test, proba)
        roc_auc = roc_auc_score(y_test, proba)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        fp_rate   = fp / (fp + tn + 1e-9)

        rows.append({
            "Model":     name,
            "PR-AUC":    round(pr_auc, 4),
            "ROC-AUC":   round(roc_auc, 4),
            "Precision": round(precision, 4),
            "Recall":    round(recall, 4),
            "F1":        round(f1, 4),
            "FP Rate":   round(fp_rate, 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "_proba":    proba,   # kept for plotting, not printed
        })

    results = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
    return results


def false_positive_reduction(results: pd.DataFrame) -> None:
    """Print false-positive reduction of best model vs LR baseline."""
    lr_fp  = results.loc[results["Model"] == "LogisticRegression", "FP"].values[0]
    best   = results.iloc[0]
    best_fp = best["FP"]
    reduction = (lr_fp - best_fp) / lr_fp * 100
    print(f"Best model : {best['Model']}  (PR-AUC={best['PR-AUC']})")
    print(f"LR baseline FP : {lr_fp:,}")
    print(f"Best model FP  : {best_fp:,}")
    print(f"False-positive reduction : {reduction:.1f}%")
