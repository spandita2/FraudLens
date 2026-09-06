"""
eda.py — Exploratory Data Analysis / data-quality inspection.

Run standalone: python src/data/eda.py
Prints a report; does not modify data. Findings from this script directly
drove decisions in preprocessing.py and feature_engineering.py.
"""
import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "transactions.csv"


def run_eda(path: Path = RAW_PATH) -> None:
    df = pd.read_csv(path)
    print("=" * 70)
    print("FRAUDLENS — DATA QUALITY & EDA REPORT")
    print("=" * 70)

    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumns and dtypes:\n{df.dtypes}")

    print("\n--- Target: isFraud ---")
    counts = df["isFraud"].value_counts()
    rate = df["isFraud"].mean()
    print(counts)
    print(f"Fraud rate: {rate:.4%}  (imbalance ratio ~ 1 fraud : {int(1/rate):,} legit)")
    print("=> Severe class imbalance. Plan: class_weight='balanced' / SMOTE on "
          "train split only, and evaluate with PR-AUC/F1/recall, NOT accuracy.")

    print("\n--- Missing values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(missing if len(missing) else "None found.")

    print("\n--- Duplicate rows ---")
    dupes = df.duplicated().sum()
    print(f"{dupes:,} fully duplicated rows ({dupes/len(df):.3%})")

    print("\n--- Transaction type breakdown ---")
    print(df["type"].value_counts())
    print("\nFraud only occurs within these types:")
    print(df.loc[df["isFraud"] == 1, "type"].value_counts())
    print("=> Confirms known PaySim property: fraud is restricted to "
          "TRANSFER and CASH_OUT. This is a real, dataset-level pattern, "
          "not leakage — it's legitimate for the model to learn 'type' as "
          "a strong feature.")

    print("\n--- Potential DATA LEAKAGE checks ---")
    print("1) isFlaggedFraud: a rule-based flag PaySim ships with "
          "(TRANSFER > 200,000). It is derived from the same event as the "
          "label using a near-perfect heuristic and would leak target "
          "information disguised as a feature.")
    print(f"   isFlaggedFraud==1 count: {(df['isFlaggedFraud']==1).sum()}, "
          f"of which fraud: {df.loc[df['isFlaggedFraud']==1, 'isFraud'].sum()}")
    print("   DECISION: isFlaggedFraud is EXCLUDED from model features.")

    print("\n2) newbalanceOrig / newbalanceDest are recorded AFTER the "
          "transaction completes. In a real-time fraud-scoring API the "
          "prediction must happen BEFORE the transaction is allowed to "
          "settle, so these post-transaction balances would not exist yet "
          "at inference time for a blocked transaction.")
    print("   DECISION: raw post-transaction balances are not used directly "
          "as features. Instead we derive pre-transaction-only signals "
          "(oldbalanceOrg, the expected vs actual balance delta computed "
          "from amount, and error terms) which are legitimate because they "
          "only require information known before/at the moment of the "
          "transaction request. This mirrors the well-documented PaySim "
          "leakage discussion in the public Kaggle notebooks for this "
          "dataset.")

    print("\n3) nameOrig / nameDest: high-cardinality identifiers, not used "
          "as raw categorical features (would not generalize / leak "
          "identity). Used only to compute aggregated frequency features.")

    print("\n--- Numeric summary: amount ---")
    print(df["amount"].describe())

    print("\nEDA complete.")


if __name__ == "__main__":
    run_eda()
