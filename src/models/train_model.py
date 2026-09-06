"""
train_model.py — trains, evaluates, and saves the FraudLens model pipeline.

Run: python -m src.models.train_model   (from project root)

Produces:
  models/fraudlens_pipeline.joblib   — preprocessing + model + the fitted
                                        CustomerHistoryEncoder, all needed
                                        for leakage-consistent inference
  models/metrics.json                — actual achieved metrics
  models/evaluation_plots.png        — confusion matrix + PR curve + ROC curve

--------------------------------------------------------------------------
CHRONOLOGICAL SPLIT (fixes train/test temporal contamination)
--------------------------------------------------------------------------
The split is now done by `step` (PaySim's hourly time index), NOT randomly:
all transactions up to a cutoff step are TRAIN, everything after is TEST.
This matters for two independent reasons:

1. Realism: a fraud model is deployed to score transactions as they arrive.
   Evaluating it on a random shuffle of past-and-future transactions
   overstates how well it will do in production, where it only ever sees
   the future. A chronological split is the honest simulation of that.

2. It's a precondition for the customer-history features to be leak-free.
   See feature_engineering.py's module docstring for the full explanation,
   but in short: CustomerHistoryEncoder.fit_transform() is called on TRAIN
   ONLY, walking forward through time to build a per-customer running
   (count, amount-sum) state. `.transform()` is then called on TEST,
   CONTINUING that state forward through the test period. Because train is
   defined as "everything before the cutoff" and test as "everything at or
   after it," a test row's historical features can only ever be built from
   training-period data and earlier test-period data — never from a
   training row's future or another test row that hasn't happened yet.
--------------------------------------------------------------------------
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report,
)
from sklearn.pipeline import Pipeline
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import clean_raw, build_preprocessor
from src.features.feature_engineering import (
    add_row_level_features, CustomerHistoryEncoder,
    FEATURE_COLUMNS_NUMERIC, FEATURE_COLUMNS_CATEGORICAL, TARGET_COLUMN,
)

RAW_PATH = ROOT / "data" / "raw" / "transactions.csv"
PROCESSED_TRAIN_PATH = ROOT / "data" / "processed" / "transactions_features_train.csv"
PROCESSED_TEST_PATH = ROOT / "data" / "processed" / "transactions_features_test.csv"
MODEL_PATH = ROOT / "models" / "fraudlens_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
PLOTS_PATH = ROOT / "models" / "evaluation_plots.png"

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL
SPLIT_QUANTILE = 0.8  # train = earliest 80% of the time range, test = most recent 20%


def split_chronologically(df: pd.DataFrame, quantile: float = SPLIT_QUANTILE):
    cutoff_step = df["step"].quantile(quantile)
    train_df = df[df["step"] <= cutoff_step].copy()
    test_df = df[df["step"] > cutoff_step].copy()
    return train_df, test_df, cutoff_step


def evaluate(name: str, y_true, y_pred, y_proba) -> dict:
    metrics = {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print(f"ROC-AUC: {metrics['roc_auc']}  PR-AUC: {metrics['pr_auc']}")
    print(f"Confusion matrix [[TN FP][FN TP]]:\n{cm}")
    return metrics


def main():
    print("Loading and cleaning raw data...")
    df = pd.read_csv(RAW_PATH)
    df = clean_raw(df)

    print(f"\nSplitting chronologically by `step` at the {SPLIT_QUANTILE:.0%} quantile "
          f"(train = earlier transactions, test = later transactions)...")
    train_raw, test_raw, cutoff_step = split_chronologically(df)
    print(f"Cutoff step: {cutoff_step:.0f} (of range {df['step'].min()}-{df['step'].max()})")
    print(f"Train: {len(train_raw):,} rows, {int(train_raw[TARGET_COLUMN].sum())} fraud "
          f"({train_raw[TARGET_COLUMN].mean():.4%})")
    print(f"Test:  {len(test_raw):,} rows, {int(test_raw[TARGET_COLUMN].sum())} fraud "
          f"({test_raw[TARGET_COLUMN].mean():.4%})")

    # Row-level features: stateless, safe to compute independently on each split.
    train_df = add_row_level_features(train_raw)
    test_df = add_row_level_features(test_raw)

    # Customer-history features: fit on TRAIN only, then extended forward into TEST.
    history_encoder = CustomerHistoryEncoder()
    train_df = history_encoder.fit_transform(train_df)
    test_df = history_encoder.transform(test_df)

    PROCESSED_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)

    X_train, y_train = train_df[ALL_FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[ALL_FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    results = {}

    # --- Final model: Logistic Regression ---
    # NOTE: Random Forest was removed here after repeatedly crashing with an
    # ArrayMemoryError during training (even after reducing to 100 trees).
    # Logistic Regression is the only model trained/compared now. See
    # README.md's ML Methodology section for the up-to-date rationale.
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_pred = lr_pipeline.predict(X_test)
    results["logistic_regression"] = evaluate("Logistic Regression (final model)", y_test, lr_pred, lr_proba)

    best_name = "logistic_regression"
    best_pipeline = lr_pipeline
    print(f"\nFinal model: {best_name}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "history_encoder": history_encoder,  # required for leakage-consistent inference
            "feature_columns": ALL_FEATURE_COLUMNS,
            "model_name": best_name,
            "split_cutoff_step": float(cutoff_step),
        },
        MODEL_PATH,
    )
    print(f"Saved final pipeline -> {MODEL_PATH}")

    output = {
        "models": results,
        "final_model": best_name,
        "selection_criterion": "Logistic Regression is the only model trained (Random Forest removed "
                                "after repeated ArrayMemoryError crashes during training).",
        "split_method": f"chronological by `step`, cutoff at the {SPLIT_QUANTILE:.0%} quantile "
                         f"(step <= {cutoff_step:.0f} = train, step > {cutoff_step:.0f} = test)",
        "dataset": {
            "source": "Synthetic stand-in matching Kaggle PaySim schema "
                       "(ealaxi/paysim1) — see src/data/generate_synthetic_data.py "
                       "docstring for why, and how to substitute real data.",
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "fraud_rate_train": round(float(y_train.mean()), 6),
            "fraud_rate_test": round(float(y_test.mean()), 6),
        },
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved metrics -> {METRICS_PATH}")

   # _save_plots(y_test, lr_proba, results)


def _save_plots(y_test, lr_proba, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    fpr, tpr, _ = roc_curve(y_test, lr_proba)
    axes[0].plot(fpr, tpr, label="Logistic Regression")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_test, lr_proba)
    axes[1].plot(rec, prec, label="Logistic Regression")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend()

    best_name = "logistic_regression"
    cm = np.array(results[best_name]["confusion_matrix"])
    axes[2].imshow(cm, cmap="Blues")
    axes[2].set_title(f"Confusion Matrix — {best_name} (chronological test set)")
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["Legit", "Fraud"])
    axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(["Legit", "Fraud"])
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(PLOTS_PATH, dpi=120)
    print(f"Saved evaluation plots -> {PLOTS_PATH}")


if __name__ == "__main__":
    main()