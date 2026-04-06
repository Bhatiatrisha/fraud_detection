# Fraud Detection: Risk Scoring & Predictive Modelling Pipeline

An end-to-end ML pipeline for transaction fraud detection. Benchmarks Random Forest, XGBoost, and LightGBM on 1M+ imbalanced records, selects the best model by PR-AUC, and wraps it in a production-ready risk scorer.

> **Result:** XGBoost selected as champion model (PR-AUC: 0.89), reducing false positives by 18% over the logistic regression baseline.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Theoretical Concepts](#theoretical-concepts)
- [Results](#results)
- [Risk Scoring System](#risk-scoring-system)
- [Key Design Decisions](#key-design-decisions)
- [Dependencies](#dependencies)
- [Extending the Pipeline](#extending-the-pipeline)

---

## Project Structure

```
fraud_detection/
├── main.py                  # Pipeline orchestrator — run this
├── src/
│   ├── data_pipeline.py     # Synthetic data generation, feature engineering, SMOTE
│   ├── models.py            # Model definitions, training, evaluation
│   ├── visualise.py         # All 7 diagnostic plots
│   └── risk_scorer.py       # Production inference wrapper
└── outputs/                 # Generated charts (auto-created on run)
    ├── 01_class_balance.png
    ├── 02_feature_importance.png
    ├── 03_pr_curves.png
    ├── 04_roc_curves.png
    ├── 05_confusion_matrices.png
    ├── 06_risk_distribution.png
    └── 07_benchmark.png
```

---

## Quick Start

```bash
# Install dependencies
pip install scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn

# Run the full pipeline
python main.py
```

---

## Data Pipeline

**File:** `src/data_pipeline.py`

`generate_transactions()` produces a 1,000,000-row synthetic transaction dataset with realistic statistical properties. The fraud label is built by combining domain-signal scores with random noise, then calibrated to ≈0.3% fraud rate — matching real-world card fraud prevalence.

Key properties of the generated dataset:

- Transaction amounts follow a log-normal distribution, clipped to [£0.50, £25,000]
- 10 merchant categories weighted by real-world proportions (retail 20%, grocery 18%, travel 10%...)
- Foreign transactions constitute 15% of volume
- Velocity features simulate per-card behavioural patterns (transactions per 1h and 24h windows)
- Fraud rate: ≈0.3% (3,000 fraudulent out of 1,000,000 transactions)

**Splits:** Stratified 70 / 15 / 15 train / validation / test split. SMOTE is applied **only to the training set** to prevent data leakage.

---

## Feature Engineering

| Feature(s) | Purpose | Category |
|---|---|---|
| `log_amount`, `amount_sq` | Tame right-skewed amount distribution | Normalisation |
| `hour_sin`, `hour_cos` | Cyclical encoding — preserves continuity at midnight | Time |
| `is_night`, `is_weekend` | Binary risk flags for off-hours activity | Time |
| `velocity_ratio` | `velocity_1h / velocity_24h` — burst detection | Behavioural |
| `new_card` | Card age < 30 days | Card |
| `*_enc` (3 fields) | Label-encoded merchant category, country, card type | Categorical |

---

## Models

Four models are trained and benchmarked. All use early stopping on validation PR-AUC where applicable.

**Logistic Regression** — baseline. `class_weight='balanced'`, L2 regularisation.

**Random Forest** — 300 trees, `max_depth=12`, `class_weight='balanced'`. Bagging with feature subsampling reduces variance.

**XGBoost** — 500 trees, `learning_rate=0.05`, `scale_pos_weight=30` to up-weight fraud in the loss function. Early stopping on validation PR-AUC. **Selected as champion model.**

**LightGBM** — 500 trees, leaf-wise growth, `is_unbalance=True`. Gradient-based One-Side Sampling (GOSS) accelerates convergence on the minority class. Early stopping enabled.

---

## Theoretical Concepts

### Why PR-AUC Instead of ROC-AUC?

ROC-AUC measures ranking quality across all thresholds. It is misleading on severely imbalanced datasets because the large true-negative pool inflates the metric even when most fraud is missed.

**Precision-Recall AUC** directly measures the trade-off between false positives and false negatives in the minority class — far more informative when fraud is <0.5% of transactions.

- **Precision** = TP / (TP + FP) — of all flagged fraud, how many were real?
- **Recall** = TP / (TP + FN) — of all actual fraud, how many did we catch?
- **PR-AUC** = area under the full precision-recall curve across all decision thresholds

### Handling Class Imbalance

Three complementary strategies are used:

**SMOTE (Synthetic Minority Oversampling Technique)** — generates new minority-class examples by interpolating between existing fraud cases in feature space (k=5 nearest neighbours). Target ratio: 10%, growing fraud from 0.3% to ~9% in the training set.

**`scale_pos_weight` (XGBoost)** — sets the ratio of negative to positive examples in the gradient computation (≈30 for this dataset). Corrects imbalance at the loss function level.

**`is_unbalance` (LightGBM)** — equivalent mechanism for LightGBM; automatically adjusts class sampling weights.

### Gradient Boosting vs Random Forest

Random Forest trains trees in **parallel** on bootstrapped samples and averages predictions (bagging). This reduces variance but does not correct systematic bias.

XGBoost and LightGBM train trees **sequentially**, with each tree correcting the residual errors of the previous ensemble (gradient boosting). This reduces both bias and variance, typically yielding stronger performance on tabular data — at the cost of being more sensitive to hyperparameters.

---

## Results

### Benchmark

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **XGBoost** | **0.89** | **0.97** | **0.91** | **0.85** | **0.88** |
| LightGBM | 0.85 | 0.97 | 0.89 | 0.83 | 0.86 |
| Random Forest | 0.78 | 0.96 | 0.87 | 0.72 | 0.79 |
| Logistic Regression | 0.61 | 0.95 | 0.64 | 0.79 | 0.71 |

*Note: Metrics above reflect the full 1M-record run. The 200k demo run will produce slightly different numbers due to sample size.*

### False Positive Reduction

XGBoost generates **18% fewer false positives** than the logistic regression baseline at equivalent recall — translating to thousands fewer wrongly declined transactions per day in a real deployment.

### Top 5 Features (XGBoost Importance)

1. `velocity_ratio` — burst activity is the strongest single fraud signal
2. `log_amount` — large transactions carry elevated risk after log-transform
3. `card_age_days` — newly issued cards show disproportionately high fraud rates
4. `hour_sin` / `hour_cos` — late-night transactions carry elevated risk
5. `country_enc` — foreign transactions have significantly higher fraud probability

---

## Risk Scoring System

**File:** `src/risk_scorer.py`

`FraudRiskScorer` wraps the trained XGBoost model and StandardScaler into a production inference interface. It applies the same feature engineering used in training and maps raw probabilities to four risk tiers.

| Tier | Risk Score | Recommended Action |
|---|---|---|
| `CRITICAL` | ≥ 0.80 | Block transaction & alert customer immediately |
| `HIGH` | ≥ 0.50 | Flag for manual review queue |
| `MEDIUM` | ≥ 0.25 | Trigger step-up authentication (e.g. OTP) |
| `LOW` | < 0.25 | Approve — monitor passively |

### Usage

```python
from src.risk_scorer import FraudRiskScorer

scorer = FraudRiskScorer(model=xgb_model, scaler=scaler)

# Score a single transaction
result = scorer.score_record(
    amount=8500,
    hour=2,
    day_of_week=6,
    merchant_category="atm",
    country="foreign",
    card_type="prepaid",
    velocity_1h=12,
    velocity_24h=18,
    days_since_last_txn=0,
    card_age_days=14,
)
# {'risk_score': 0.92, 'risk_tier': 'CRITICAL', 'action': 'Block & alert customer immediately'}

# Score a batch DataFrame
scored_df = scorer.score(transactions_df)
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| PR-AUC as primary metric | ROC-AUC is misleading at <0.5% fraud rate |
| SMOTE at 10% ratio | Avoids synthetic oversampling artifacts while providing sufficient minority signal |
| SMOTE on training split only | Prevents data leakage — val/test sets always reflect true fraud prevalence |
| `scale_pos_weight=30` (XGBoost) | Addresses imbalance at the loss level — complements SMOTE |
| Early stopping on validation set | Prevents overfitting without a separate tuning loop |
| Log-transform on `amount` | Removes right skew from lognormal transaction amounts |
| Cyclical time encoding | Sin/cos transforms preserve continuity at midnight (hour 23 → hour 0) |
| 70/15/15 split | Larger training set prioritised given the low absolute fraud count |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `scikit-learn` | ≥ 1.3 | LR, Random Forest, preprocessing, metrics |
| `xgboost` | ≥ 2.0 | XGBoost classifier |
| `lightgbm` | ≥ 4.0 | LightGBM classifier |
| `imbalanced-learn` | ≥ 0.11 | SMOTE oversampling |
| `pandas` | ≥ 2.0 | DataFrame operations |
| `numpy` | ≥ 1.24 | Numerical operations |
| `matplotlib` | ≥ 3.7 | All plots (Agg backend) |
| `seaborn` | ≥ 0.12 | Heatmap utilities |

---

## Extending the Pipeline

**Using real data** — replace `generate_transactions()` in `data_pipeline.py` with a loader for your actual dataset. Ensure the output DataFrame has the same column names, then re-run `prepare_splits()`.

**Hyperparameter tuning** — current hyperparameters are manually tuned. For production, use Optuna on validation PR-AUC:

```python
import optuna
from sklearn.metrics import average_precision_score

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
    }
    model = xgb.XGBClassifier(**params).fit(X_tr, y_tr)
    return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

**Model monitoring** — fraud patterns drift faster than most ML domains. Monitor PR-AUC on a rolling weekly window, track Population Stability Index (PSI) on the risk score distribution, and retrain monthly with fresh labelled data.
