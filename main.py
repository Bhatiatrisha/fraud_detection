"""
main.py
-------
End-to-end pipeline runner.

Usage:
    python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from src.data_pipeline  import generate_transactions, prepare_splits, FEATURE_COLS
from src.models         import train_all, evaluate, false_positive_reduction
from src.visualise      import (
    plot_class_balance, plot_feature_importance, plot_pr_curves,
    plot_roc_curves, plot_confusion_matrices, plot_risk_distribution,
    plot_benchmark,
)
from src.risk_scorer    import FraudRiskScorer, demo_scoring


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # ── 1. DATA ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  FRAUD DETECTION · Risk Scoring & Predictive ML Pipeline")
    print("=" * 60)

    print("\n▶ Generating 1 M synthetic transactions…")
    df = generate_transactions(n=1_000_000)
    fraud_rate = df["is_fraud"].mean() * 100
    print(f"  Total rows : {len(df):,}")
    print(f"  Fraud rate : {fraud_rate:.3f}%  ({df['is_fraud'].sum():,} fraudulent)")

    plot_class_balance(df["is_fraud"], f"{OUTPUT_DIR}/01_class_balance.png")
    print("  ✓ Saved: 01_class_balance.png")

    # ── 2. PREPROCESSING ─────────────────────────────────────────────────────
    print("\n▶ Preprocessing & SMOTE oversampling…")
    X_tr, X_val, X_test, y_tr, y_val, y_test, scaler = prepare_splits(df)

    # ── 3. TRAINING ──────────────────────────────────────────────────────────
    print("\n▶ Training models…")
    models = train_all(X_tr, X_val, y_tr, y_val)

    # ── 4. EVALUATION ────────────────────────────────────────────────────────
    print("▶ Evaluating on held-out test set…\n")
    results = evaluate(models, X_test, y_test)

    display_cols = ["Model","PR-AUC","ROC-AUC","Precision","Recall","F1","FP Rate"]
    print(results[display_cols].to_string(index=False))
    print()
    false_positive_reduction(results)

    # ── 5. PLOTS ─────────────────────────────────────────────────────────────
    print("\n▶ Generating plots…")
    plot_feature_importance(
        models["XGBoost"], FEATURE_COLS,
        f"{OUTPUT_DIR}/02_feature_importance.png"
    )
    plot_pr_curves(results, y_test,    f"{OUTPUT_DIR}/03_pr_curves.png")
    plot_roc_curves(results, y_test,   f"{OUTPUT_DIR}/04_roc_curves.png")
    plot_confusion_matrices(results, y_test, f"{OUTPUT_DIR}/05_confusion_matrices.png")
    plot_risk_distribution(
        models["XGBoost"], X_test, y_test,
        f"{OUTPUT_DIR}/06_risk_distribution.png"
    )
    plot_benchmark(results,             f"{OUTPUT_DIR}/07_benchmark.png")
    print("  ✓ All 7 plots saved to outputs/")

    # ── 6. RISK SCORER DEMO ──────────────────────────────────────────────────
    print("\n▶ Risk Scorer — live demo (8 sample transactions):")
    scorer = FraudRiskScorer(models["XGBoost"], scaler)
    demo_df = demo_scoring(scorer)
    print(demo_df.to_string(index=False))

    # ── 7. SUMMARY ──────────────────────────────────────────────────────────
    best = results.iloc[0]
    lr   = results[results["Model"] == "LogisticRegression"].iloc[0]
    fp_reduction = (lr["FP"] - best["FP"]) / lr["FP"] * 100

    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Dataset       : 1,000,000 transactions  (fraud ≈ 0.3%)")
    print(f"  Best model    : {best['Model']}")
    print(f"  PR-AUC        : {best['PR-AUC']}")
    print(f"  ROC-AUC       : {best['ROC-AUC']}")
    print(f"  FP reduction  : {fp_reduction:.1f}% vs LR baseline")
    print("=" * 60)

    return results, models, scaler


if __name__ == "__main__":
    results, models, scaler = main()
