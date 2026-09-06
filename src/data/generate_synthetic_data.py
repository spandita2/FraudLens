"""
generate_synthetic_data.py
---------------------------
IMPORTANT — READ THIS FIRST:

This script generates a SYNTHETIC stand-in dataset that reproduces the exact
column schema, data types, and approximate statistical properties (fraud
rate, transaction-type mix, amount distribution) of the real public dataset:

    "Synthetic Financial Datasets For Fraud Detection" (PaySim)
    Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1
    ~6.36M rows, 11 columns, isFraud target, ~0.13% fraud rate.

It exists ONLY because this build environment has no internet access to
download the real file. It is NOT the real dataset and must not be
represented as such. Every metric reported later in this project (model
scores, dashboard KPIs, README numbers) is computed on THIS synthetic data.

TO USE THE REAL DATASET INSTEAD (recommended before treating this as a
portfolio piece):
    1. pip install kaggle
    2. Place your kaggle.json API credentials in ~/.kaggle/
    3. kaggle datasets download -d ealaxi/paysim1
    4. unzip paysim1.zip -d data/raw/
    5. Rename the CSV to data/raw/transactions.csv (same column names)
    6. Re-run: python src/data/run_pipeline.py
   No other code changes are needed — every downstream script only assumes
   the PaySim column schema below, not that the data is synthetic.

PaySim column schema (reproduced exactly):
    step            int    — 1 unit = 1 hour of simulated time (1..743)
    type            str    — CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    amount          float  — transaction amount (local currency)
    nameOrig        str    — customer who started the transaction
    oldbalanceOrg   float  — initiator balance before transaction
    newbalanceOrig  float  — initiator balance after transaction
    nameDest        str    — recipient
    oldbalanceDest  float  — recipient balance before transaction
    newbalanceDest  float  — recipient balance after transaction
    isFraud         int    — target (1 = fraud), known in real data to occur
                             only within TRANSFER and CASH_OUT types
    isFlaggedFraud  int    — a naive rule-based flag PaySim includes for
                             attempted transfers over 200,000 (kept here for
                             schema fidelity; NOT used as a model feature —
                             see leakage note in eda.py)
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_ROWS = 200_000  # scaled down from PaySim's 6.36M for fast local iteration
FRAUD_RATE = 0.0013  # matches real PaySim's ~0.13% fraud rate
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "transactions.csv"

TYPE_PROBS = {
    "CASH_IN": 0.22,
    "CASH_OUT": 0.35,
    "DEBIT": 0.01,
    "PAYMENT": 0.34,
    "TRANSFER": 0.08,
}


def _customer_id(prefix: str, rng: np.random.Generator) -> str:
    return f"{prefix}{rng.integers(10_000_000, 99_999_999)}"


def generate(n_rows: int = N_ROWS, fraud_rate: float = FRAUD_RATE, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    types = rng.choice(list(TYPE_PROBS.keys()), size=n_rows, p=list(TYPE_PROBS.values()))
    steps = rng.integers(1, 744, size=n_rows)  # 743 hours ~ 31 simulated days, matches PaySim range

    # Real fraud in PaySim occurs only in TRANSFER / CASH_OUT. Reproduce that constraint.
    fraud_eligible = np.isin(types, ["TRANSFER", "CASH_OUT"])
    is_fraud = np.zeros(n_rows, dtype=int)
    n_target_fraud = int(n_rows * fraud_rate)
    eligible_idx = np.where(fraud_eligible)[0]
    fraud_idx = rng.choice(eligible_idx, size=min(n_target_fraud, len(eligible_idx)), replace=False)
    is_fraud[fraud_idx] = 1

    amount = np.empty(n_rows)
    old_orig = np.empty(n_rows)
    new_orig = np.empty(n_rows)
    old_dest = np.empty(n_rows)
    new_dest = np.empty(n_rows)
    name_orig = np.array([_customer_id("C", rng) for _ in range(n_rows)], dtype=object)
    name_dest = np.empty(n_rows, dtype=object)

    for i in range(n_rows):
        if is_fraud[i] == 1:
            # Fraud pattern reproduced from PaySim's documented behavior:
            # the fraudster empties the origin account (oldbalance -> ~0)
            # and the destination account balance is often left unchanged
            # (i.e. the recorded newbalanceDest doesn't reflect the credit) —
            # this asymmetry is the classic PaySim fraud signature.
            old_orig[i] = rng.uniform(1_000, 500_000)
            amount[i] = old_orig[i] * rng.uniform(0.9, 1.0)
            new_orig[i] = max(old_orig[i] - amount[i], 0.0)
            old_dest[i] = rng.uniform(0, 50_000)
            new_dest[i] = old_dest[i]  # unchanged -> leakage-relevant signature
            name_dest[i] = _customer_id("C", rng)
        else:
            amount[i] = np.round(rng.lognormal(mean=6.5, sigma=1.6), 2)
            old_orig[i] = max(rng.lognormal(mean=8.0, sigma=1.8) - amount[i], 0.0)
            if types[i] in ("CASH_IN",):
                new_orig[i] = old_orig[i] + amount[i]
            else:
                new_orig[i] = max(old_orig[i] - amount[i], 0.0)
            if types[i] == "PAYMENT":
                # PaySim convention: destination balances are not tracked for merchants
                old_dest[i] = 0.0
                new_dest[i] = 0.0
                name_dest[i] = "M" + str(rng.integers(10_000_000, 99_999_999))
            else:
                old_dest[i] = rng.lognormal(mean=7.5, sigma=1.8)
                new_dest[i] = old_dest[i] + amount[i]
                name_dest[i] = _customer_id("C", rng)

    is_flagged = ((types == "TRANSFER") & (amount > 200_000)).astype(int)

    df = pd.DataFrame({
        "step": steps,
        "type": types,
        "amount": np.round(amount, 2),
        "nameOrig": name_orig,
        "oldbalanceOrg": np.round(old_orig, 2),
        "newbalanceOrig": np.round(new_orig, 2),
        "nameDest": name_dest,
        "oldbalanceDest": np.round(old_dest, 2),
        "newbalanceDest": np.round(new_dest, 2),
        "isFraud": is_fraud,
        "isFlaggedFraud": is_flagged,
    })

    # Inject realistic data-quality issues so the cleaning step in the
    # pipeline has real work to do (PaySim itself is fairly clean, but any
    # production feed has some of this):
    dup_idx = rng.choice(n_rows, size=int(n_rows * 0.002), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)  # duplicate rows
    null_idx = rng.choice(df.index, size=int(len(df) * 0.001), replace=False)
    df.loc[null_idx, "oldbalanceDest"] = np.nan  # missing values

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote synthetic PaySim-schema dataset: {OUT_PATH}")
    print(f"Rows: {len(df):,} | Fraud rate: {df['isFraud'].mean():.4%}")
    print(df.dtypes)
