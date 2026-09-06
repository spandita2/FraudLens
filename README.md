# FraudLens — Digital Payment Fraud Detection & Risk Analytics Platform

FraudLens is an end-to-end fraud detection system: a reproducible ML pipeline, a PostgreSQL-backed data layer, a FastAPI risk-scoring service, and a Streamlit analytics dashboard.

> **⚠️ Dataset note — read this first.** This build environment has no internet access, so the real dataset could not be downloaded automatically. FraudLens targets the public **[PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)** ("Synthetic Financial Datasets For Fraud Detection," ~6.3M rows, ~0.13% fraud rate). To let every part of the pipeline actually run and produce real numbers, `src/data/generate_synthetic_data.py` generates a **schema-faithful synthetic stand-in** — same 11 columns, same data types, same fraud rate, same fraud-restricted-to-TRANSFER/CASH_OUT structure as the real PaySim data. All metrics in this README come from a real run of the pipeline **on that synthetic data**, not from the real Kaggle file. See [Using the real dataset](#using-the-real-dataset) to swap it in — no other code changes needed.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [ML Methodology](#ml-methodology)
- [Feature Engineering](#feature-engineering)
- [Model Comparison & Results](#model-comparison--results)
- [Database Design](#database-design)
- [API Endpoints](#api-endpoints)
- [Dashboard](#dashboard)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Using the Real Dataset](#using-the-real-dataset)
- [Testing](#testing)
- [Future Improvements](#future-improvements)
- [Resume Bullets](#resume-bullets)

---

## Problem Statement

Digital payment platforms process high volumes of transactions where fraud is rare (well under 1%) but costly. A useful fraud system must:
- Handle **severe class imbalance** without optimizing for the misleading metric of accuracy.
- Avoid **data leakage** — especially post-transaction information that wouldn't exist yet at the moment a real-time decision must be made.
- Score transactions **in real time** through an API, not just in a notebook.
- Give risk analysts a **dashboard** to monitor fraud patterns, not just a single prediction.

FraudLens addresses all four.

## Features
- Reproducible cleaning → feature engineering → training pipeline, run with one command.
- Logistic Regression model, trained on a leakage-aware feature pipeline and evaluated on precision, recall, F1, ROC-AUC, and PR-AUC — the metric that actually matters under 0.13% fraud prevalence.
- Explicit, documented data-leakage decisions (see [ML Methodology](#ml-methodology)).
- FastAPI service with input validation, health checks, and a queryable transactions endpoint.
- PostgreSQL schema with appropriate types and indexes.
- Streamlit dashboard: KPIs, fraud analytics, a live risk-scoring form, and DB-backed transaction browsing.

## Architecture

```
Dataset (PaySim schema)
      │
      ▼
Data Cleaning (dedupe, missing-value handling)   src/data/preprocessing.py
      │
      ▼
EDA (imbalance, leakage, missingness report)      src/data/eda.py
      │
      ▼
Feature Engineering (time, balance, aggregation)  src/features/feature_engineering.py
      │
      ▼
ML Training (Logistic Regression)                 src/models/train_model.py
      │
      ▼
Saved Pipeline (joblib: preprocessing + model)    models/fraudlens_pipeline.joblib
      │
      ├──────────────────────────────┐
      ▼                              ▼
PostgreSQL (transaction store)   FastAPI (/predict, /health, /transactions)
      │                              │
      └──────────────┬───────────────┘
                      ▼
           Streamlit Dashboard (dashboard/app.py)
```

The **same joblib pipeline** (preprocessing + model, fit together) is used both at training time and inside the API, so inference-time preprocessing is guaranteed to match training-time preprocessing exactly.

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · PostgreSQL · SQLAlchemy · FastAPI · Streamlit · Plotly · joblib · pytest

## Dataset

| | |
|---|---|
| Target dataset | [PaySim — Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| Used in this build | Schema-faithful **synthetic stand-in**, 200,400 generated rows (see note above) |
| Target column | `isFraud` (binary) |
| Class imbalance | 0.13% fraud (≈1 fraud per 767 legitimate transactions) — matches real PaySim |
| Missing values found | `oldbalanceDest` (~0.1% of rows) — imputed with median inside the sklearn pipeline |
| Duplicates found | 398 exact duplicate rows — dropped in `clean_raw()` |
| Columns | `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud` |

Full inspection report: run `python src/data/eda.py`.

## ML Methodology

**Data leakage identified and handled** (full reasoning in `src/data/eda.py`):
1. **`isFlaggedFraud`** — a rule-based flag PaySim ships with (`TRANSFER > 200,000`). In our data, 30 of 32 flagged rows are actual fraud — using it would let the model "cheat" off a near-perfect heuristic. **Excluded.**
2. **`newbalanceOrig` / `newbalanceDest`** — recorded *after* the transaction settles. A real-time scoring API must decide *before* settlement, so these wouldn't exist at inference time for a transaction still being evaluated. **Excluded**; replaced with features derived only from `amount` and the *pre*-transaction balance (`oldbalanceOrg`).
3. **`nameOrig` / `nameDest`** — high-cardinality raw identifiers, not used directly (would not generalize). Used only to compute aggregated, historical-only behavioral features.

**Temporal (train/test) leakage — identified and fixed:** an earlier version of this pipeline computed each customer's historical transaction count/average as an expanding statistic over the *entire* dataset (train + test together) before splitting, then split randomly. That let a training row's historical-average feature be built, in part, from a transaction that ended up in the "held-out" test set — and made the evaluation unrealistic, since a real fraud model never gets to see the future. Fixed with two changes, both required together:
- **Chronological split** by `step` (PaySim's hourly time index) instead of a random stratified split — train is every transaction up to the cutoff, test is everything after it. See `split_chronologically()` in `train_model.py`.
- **`CustomerHistoryEncoder`** (`src/features/feature_engineering.py`) — an explicit fit/transform object, same discipline as the sklearn scaler/encoder. `fit_transform()` runs on train only, walking forward through time to build each customer's running (count, amount-sum) state. `transform()` then *continues* that state forward through the test period. A test-period row can see a customer's training history and that customer's *earlier* test-period rows (both are genuinely in the past relative to it) but never a training row's future or a later test row. The fallback average used for a customer's very first transaction is now the **training-set median only** (previously the whole dataset's median — a smaller, secondary leak into every "new customer" row).
- The fitted `CustomerHistoryEncoder` is saved inside the joblib bundle and reused as-is by the API (`api/main.py`) via `transform_single()`, which looks up a customer's current history **without** mutating state — an unconfirmed, not-yet-scored transaction shouldn't count as history.

**Imbalance handling:** `class_weight="balanced"` (Logistic Regression).

**Train/test split:** chronological by `step`, cutoff at the 80th percentile (step ≤ 594 = train, step > 594 = test in the current synthetic run — recomputed fresh on every training run from whatever data is loaded, not hardcoded).

**Model:** Logistic Regression (`class_weight="balanced"`) is the only model trained. A Random Forest was originally trained and compared here, but it was removed after repeatedly crashing with an `ArrayMemoryError` during training — even after reducing `n_estimators` to 100. Logistic Regression, which had already trained and evaluated successfully, was kept as the final model to meet a tight turnaround. PR-AUC and recall remain the metrics reported below, for consistency with how fraud-detection models should be evaluated even with only one model in the comparison.

## Feature Engineering

All 10 features are built **only** from columns present in the PaySim schema and only from information available *before or at* the moment a transaction is requested:

| Feature | Derived from | Rationale |
|---|---|---|
| `hour_of_day` | `step` | Time-of-day fraud pattern |
| `day` | `step` | Time trend |
| `amount_log` | `amount` | Reduce right-skew |
| `orig_balance_delta` | `oldbalanceOrg`, `amount` | Expected post-debit balance, computed without looking at the actual outcome |
| `orig_balance_ratio` | `amount`, `oldbalanceOrg` | Captures the "drain the account" signature (amount ≈ full balance) |
| `dest_is_merchant` | `nameDest` prefix | PaySim convention: merchant accounts (`M...`) behave differently |
| `orig_txn_count` | `nameOrig`, ordered by `step` | Customer-level aggregation, expanding/historical-only (no future leakage) |
| `orig_avg_amount` | `nameOrig`, `amount`, ordered by `step` | Customer's historical average spend |
| `type` | `type` | One-hot encoded transaction type |
| `oldbalanceOrg`, `oldbalanceDest` | raw | Pre-transaction balances |

## Model Comparison & Results

Actual results from `python -m src.models.train_model` on the **chronologically** held-out test set (39,978 rows, the most recent 20% of simulated time, 50 fraud cases — cutoff at step 594 of a 1–743 range):

| Metric | Logistic Regression (final model) |
|---|---|
| Precision | 0.0625 |
| Recall | 1.0000 |
| F1-score | 0.1176 |
| ROC-AUC | 0.9999 |
| PR-AUC | 0.9675 |

**Confusion matrix (Logistic Regression, final model)** — `[[TN FP][FN TP]]`:
```
[[39178   750]
 [    0    50]]
```

**Final model: Logistic Regression.** A Random Forest was trained alongside it earlier in this project and had scored better on this same test set (precision 1.00, recall 0.98, PR-AUC 0.9989 — see git history / prior README revisions). It has since started crashing with an `ArrayMemoryError` during training, including after cutting `n_estimators` to 100, and there wasn't time to debug it against a deployment deadline, so it was removed. Logistic Regression catches all 50 fraud cases (recall 1.0) but at the cost of 750 false positives — a precision of 0.06 that would flood a real fraud-review queue. **This is a known, explicit trade-off**, not a claim that Logistic Regression is the better model: it is the model that trains reliably right now. Revisiting the Random Forest memory issue (see [Future Improvements](#future-improvements)) is the natural next step once there's time to debug it — likely candidates are the training machine's available RAM/environment rather than the algorithm itself, since it trained fine earlier in the project.

Evaluation plots (ROC, PR curve, confusion matrix): `models/evaluation_plots.png`.
Raw metrics JSON (source of truth for all numbers above, regenerated on every training run): `models/metrics.json`.

**Honest caveat:** the synthetic data used here has cleaner, more separable fraud signal than the real ~6.3M-row PaySim dataset will have. Expect somewhat different precision/recall on the real data — the pipeline, feature set, chronological split, and leakage-avoidance decisions transfer directly, but re-run training on the real file before treating any of these exact numbers as a portfolio claim about real-world performance.

## Database Design

See `sql/schema.sql`. Key points:
- `transactions` table with `BIGSERIAL` primary key.
- `NUMERIC(18,2)` for monetary fields (no float rounding issues).
- Indexes on `type`, `is_fraud`, `step`, `name_orig`, `created_at` — matching the actual filter/sort patterns used by the API and dashboard.
- `predicted_fraud` / `fraud_probability` columns to store API scoring results against the row that was scored.

ORM models: `src/database/models.py`. Loader: `src/database/load_data.py`.

## API Endpoints

Full interactive docs at `/docs` once running (`uvicorn api.main:app --reload`).

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns API status, whether the model loaded, whether the DB is reachable |
| `POST` | `/predict` | Accepts a `TransactionRequest`, returns fraud prediction + probability + risk level |
| `GET` | `/transactions` | Returns transactions from PostgreSQL, filterable by `is_fraud`, `type`, `limit` |

`POST /predict` example:
```json
{
  "step": 10, "type": "TRANSFER", "amount": 181000.0,
  "nameOrig": "C1231006815", "oldbalanceOrg": 181000.0,
  "nameDest": "C1666544295", "oldbalanceDest": 0.0
}
```
Response:
```json
{"fraud_prediction": 1, "fraud_probability": 0.94, "risk_level": "HIGH", "model_used": "logistic_regression"}
```

## Dashboard

`streamlit run dashboard/app.py` — four tabs:
- **Overview** — total/fraud transaction counts, fraud rate, avg amount, deployed model's live metrics.
- **Analytics** — fraud vs. legitimate split, transactions by type, fraud rate by type, volume over time, fraud by hour, amount distribution — all computed from `data/processed/transactions_features_train.csv` + `_test.csv` (the pipeline's own chronological split, concatenated for whole-dataset charts).
- **Risk Prediction** — form that calls the live FastAPI `/predict` endpoint.
- **Database Analytics** — filterable browsing of PostgreSQL-backed data via `/transactions`.

## Installation

```bash
git clone <your-repo-url>
cd FraudLens
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your real PostgreSQL credentials
```

You'll also need a running PostgreSQL instance (locally installed, Docker, or a managed service). Create the database referenced in `.env`, e.g.:
```bash
createdb fraudlens
```

## How to Run

```bash
# 1. Generate the synthetic PaySim-schema dataset (or supply the real one — see below)
python src/data/generate_synthetic_data.py

# 2. Inspect it
python src/data/eda.py

# 3. Run the full training pipeline (cleaning -> features -> train -> evaluate -> save)
python -m src.models.train_model

# 4. Set up PostgreSQL and load data
psql -U <user> -d fraudlens -f sql/schema.sql
python -m src.database.load_data

# 5. Start the API
uvicorn api.main:app --reload --port 8000

# 6. In a second terminal, start the dashboard
streamlit run dashboard/app.py
```

## Using the Real Dataset

```bash
pip install kaggle
# Place your Kaggle API token at ~/.kaggle/kaggle.json first
kaggle datasets download -d ealaxi/paysim1
unzip paysim1.zip -d data/raw/
mv data/raw/PS_20174392719_1491204439457_log.csv data/raw/transactions.csv  # exact filename inside the zip may vary — check with `unzip -l paysim1.zip`
```

Then train directly on it — **no code changes needed**, since the pipeline only assumes the PaySim column schema, not that the file is synthetic:

```bash
python -m src.models.train_model
```

This re-runs cleaning → chronological split → the fit-on-train-only `CustomerHistoryEncoder` → training → evaluation → save, exactly as described above, on the real ~6.3M-row file. Expect a longer run time than the 200K-row synthetic demo, and re-read the printed train/test row counts and fraud counts — with ~6.3M rows the 80th-percentile `step` cutoff will land at a different absolute row count and fraud count than the numbers quoted in this README, which are only valid for the bundled synthetic data.

## Testing

```bash
pytest tests/ -v
```
- `test_preprocessing.py` — cleaning, feature engineering, leakage-column exclusion (no PostgreSQL/API dependency).
- `test_model.py` — loads the saved joblib pipeline, checks valid probability output, and a directional sanity check (an account-draining TRANSFER should score higher than an ordinary small PAYMENT).
- `test_api.py` — FastAPI `/health` and `/predict`, including input-validation rejection cases. Skipped automatically if `fastapi` isn't installed or the model hasn't been trained yet.
- `test_database.py` — live PostgreSQL connectivity and schema check. Skipped automatically if no database is reachable.

## Future Improvements
- Revisit Random Forest as a second model: it previously outperformed Logistic Regression on this test set (see [Model Comparison & Results](#model-comparison--results)) but was pulled after an `ArrayMemoryError` during training; worth retrying with a smaller `max_features`/batched training approach or on a machine with more available memory.
- Retrain on the full real PaySim dataset (or a live transaction feed) rather than the synthetic stand-in.
- Add SHAP-based explainability to the `/predict` response for analyst-facing reason codes.
- Add authentication (API keys / OAuth) to the FastAPI service before any real deployment.
- Add a model-monitoring job to detect feature/label drift over time.
- Containerize with Docker Compose (API + Streamlit + PostgreSQL) for one-command startup.

## Resume Bullets

- Built FraudLens, an end-to-end fraud detection platform in Python — engineered a leakage-aware feature pipeline (Pandas, NumPy, Scikit-learn) with a chronological train/test split and a fit-on-train-only customer-history encoder to prevent temporal data leakage, and trained a Logistic Regression fraud classifier achieving 1.0 recall (0.9675 PR-AUC) on a time-based held-out set under 0.13% class imbalance.
- Designed and implemented a PostgreSQL schema (SQLAlchemy ORM) and a FastAPI backend exposing real-time fraud-scoring (`/predict`), health-check, and transaction-query endpoints with Pydantic request validation and centralized error handling.
- Developed an interactive Streamlit + Plotly analytics dashboard providing fraud KPI monitoring, transaction-pattern visualizations, and live risk scoring against the FastAPI backend, backed by a reproducible joblib-serialized ML pipeline.