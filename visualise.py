"""
visualise.py
------------
All plots used in the project report:
  1. Class imbalance bar
  2. Feature importance (XGBoost)
  3. PR curves for all models
  4. ROC curves for all models
  5. Confusion matrices (2×2 grid)
  6. Risk score distribution
  7. Model benchmark summary bar chart
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix, average_precision_score
)


# ── STYLE ─────────────────────────────────────────────────────────────────────

PALETTE = {
    "LogisticRegression": "#94a3b8",
    "RandomForest":       "#38bdf8",
    "XGBoost":            "#f97316",
    "LightGBM":           "#a78bfa",
}
DARK_BG  = "#0f172a"
CARD_BG  = "#1e293b"
TEXT     = "#f1f5f9"
ACCENT   = "#f97316"

def _style(fig, axes=None):
    fig.patch.set_facecolor(DARK_BG)
    if axes is not None:
        for ax in (axes if hasattr(axes, "__iter__") else [axes]):
            ax.set_facecolor(CARD_BG)
            ax.tick_params(colors=TEXT)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
            ax.title.set_color(TEXT)
            for spine in ax.spines.values():
                spine.set_color("#334155")


# ── 1. CLASS IMBALANCE ────────────────────────────────────────────────────────

def plot_class_balance(y, save_path):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Legitimate", "Fraud"], counts.values,
                  color=["#38bdf8", ACCENT], width=0.45, zorder=3)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                f"{v:,}\n({v/len(y)*100:.2f}%)",
                ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.set_title("Class Distribution (1 M transactions)", fontsize=11, pad=12)
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.2, zorder=0)
    _style(fig, ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 2. FEATURE IMPORTANCE ─────────────────────────────────────────────────────

def plot_feature_importance(xgb_model, feature_names, save_path):
    importances = pd.Series(
        xgb_model.feature_importances_, index=feature_names
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [ACCENT if i >= len(importances) - 5 else "#475569"
              for i in range(len(importances))]
    ax.barh(importances.index, importances.values, color=colors, height=0.65)
    ax.set_title("XGBoost Feature Importances", fontsize=11, pad=12)
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.2)
    _style(fig, ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 3. PR CURVES ─────────────────────────────────────────────────────────────

def plot_pr_curves(results, y_test, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in results.iterrows():
        name  = row["Model"]
        proba = row["_proba"]
        p, r, _ = precision_recall_curve(y_test, proba)
        auc     = average_precision_score(y_test, proba)
        lw      = 2.5 if name == "XGBoost" else 1.5
        ls      = "-"  if name == "XGBoost" else "--"
        ax.plot(r, p, label=f"{name}  (AUC={auc:.3f})",
                color=PALETTE[name], lw=lw, linestyle=ls)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontsize=11, pad=12)
    legend = ax.legend(fontsize=8, framealpha=0.15, labelcolor=TEXT)
    legend.get_frame().set_facecolor(CARD_BG)
    ax.grid(alpha=0.2)
    _style(fig, ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 4. ROC CURVES ─────────────────────────────────────────────────────────────

def plot_roc_curves(results, y_test, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], ":", color="#475569", lw=1)
    for _, row in results.iterrows():
        name  = row["Model"]
        proba = row["_proba"]
        fpr, tpr, _ = roc_curve(y_test, proba)
        lw = 2.5 if name == "XGBoost" else 1.5
        ax.plot(fpr, tpr, label=f"{name}  (AUC={row['ROC-AUC']:.3f})",
                color=PALETTE[name], lw=lw)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontsize=11, pad=12)
    legend = ax.legend(fontsize=8, framealpha=0.15, labelcolor=TEXT)
    legend.get_frame().set_facecolor(CARD_BG)
    ax.grid(alpha=0.2)
    _style(fig, ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 5. CONFUSION MATRICES ─────────────────────────────────────────────────────

def plot_confusion_matrices(results, y_test, save_path):
    n     = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8))
    cmap  = LinearSegmentedColormap.from_list("fraud", ["#1e293b", ACCENT])

    for ax, (_, row) in zip(axes, results.iterrows()):
        proba = row["_proba"]
        pred  = (proba >= 0.5).astype(int)
        cm    = confusion_matrix(y_test, pred)
        im    = ax.imshow(cm, cmap=cmap, aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        color=TEXT, fontsize=9, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Legit", "Pred Fraud"], fontsize=8)
        ax.set_yticklabels(["Actual Legit", "Actual Fraud"], fontsize=8)
        ax.set_title(row["Model"], fontsize=9, pad=8)
    _style(fig, list(axes))
    fig.suptitle("Confusion Matrices (threshold = 0.5)", color=TEXT, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 6. RISK SCORE DISTRIBUTION ────────────────────────────────────────────────

def plot_risk_distribution(xgb_model, X_test, y_test, save_path):
    proba = xgb_model.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 60)
    ax.hist(proba[y_test == 0], bins=bins, alpha=0.7,
            color="#38bdf8", label="Legitimate", density=True)
    ax.hist(proba[y_test == 1], bins=bins, alpha=0.85,
            color=ACCENT, label="Fraud", density=True)
    ax.axvline(0.5, color="#f8fafc", lw=1.2, ls="--", label="Threshold 0.5")
    ax.set_xlabel("Risk Score (XGBoost probability)")
    ax.set_ylabel("Density")
    ax.set_title("Risk Score Distribution", fontsize=11, pad=12)
    legend = ax.legend(fontsize=8, framealpha=0.15, labelcolor=TEXT)
    legend.get_frame().set_facecolor(CARD_BG)
    ax.grid(alpha=0.2)
    _style(fig, ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 7. BENCHMARK BAR ─────────────────────────────────────────────────────────

def plot_benchmark(results, save_path):
    metrics = ["PR-AUC", "ROC-AUC", "F1"]
    x       = np.arange(len(metrics))
    n       = len(results)
    width   = 0.18

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (_, row) in enumerate(results.iterrows()):
        vals = [row[m] for m in metrics]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=row["Model"],
                      color=PALETTE[row["Model"]], alpha=0.90, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=TEXT)

    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.08)
    ax.set_title("Model Benchmark — PR-AUC | ROC-AUC | F1", fontsize=11, pad=12)
    legend = ax.legend(fontsize=8, framealpha=0.15, labelcolor=TEXT, loc="upper right")
    legend.get_frame().set_facecolor(CARD_BG)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    _style(fig, ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
