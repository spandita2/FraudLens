import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import clean_raw, build_preprocessor
from src.features.feature_engineering import engineer_features, FEATURE_COLUMNS_NUMERIC


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "step": [1, 1, 2, 2],
        "type": ["TRANSFER", "PAYMENT", "TRANSFER", "PAYMENT"],
        "amount": [1000.0, 50.0, 1000.0, 200.0],
        "nameOrig": ["C1", "C2", "C1", "C3"],
        "oldbalanceOrg": [2000.0, 100.0, 2000.0, 500.0],
        "newbalanceOrig": [1000.0, 50.0, 1000.0, 300.0],
        "nameDest": ["C9", "M1", "C9", "M2"],
        "oldbalanceDest": [0.0, 0.0, 0.0, 0.0],
        "newbalanceDest": [1000.0, 0.0, 1000.0, 0.0],
        "isFraud": [1, 0, 1, 0],
        "isFlaggedFraud": [0, 0, 0, 0],
    })


def test_clean_raw_drops_exact_duplicates(sample_df):
    dup_df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    cleaned = clean_raw(dup_df)
    assert len(cleaned) == len(sample_df)


def test_engineer_features_creates_expected_columns(sample_df):
    out = engineer_features(sample_df)
    for col in FEATURE_COLUMNS_NUMERIC:
        assert col in out.columns, f"missing feature column: {col}"


def test_dest_is_merchant_flag(sample_df):
    out = engineer_features(sample_df)
    merchant_rows = out[out["nameDest"].str.startswith("M")]
    assert (merchant_rows["dest_is_merchant"] == 1).all()
    non_merchant_rows = out[~out["nameDest"].str.startswith("M")]
    assert (non_merchant_rows["dest_is_merchant"] == 0).all()


def test_orig_txn_count_is_historical_only(sample_df):
    out = engineer_features(sample_df)
    c1_rows = out[out["nameOrig"] == "C1"].sort_values("step")
    counts = c1_rows["orig_txn_count"].tolist()
    assert counts == sorted(counts), "txn count must be non-decreasing per customer over time"
    assert counts[0] == 0, "first transaction for a customer must have 0 prior transactions"


def test_build_preprocessor_fits_and_transforms(sample_df):
    features = engineer_features(sample_df)
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(features)
    assert X.shape[0] == len(sample_df)


def test_no_isFlaggedFraud_or_raw_ids_in_feature_columns():
    from src.features.feature_engineering import FEATURE_COLUMNS_NUMERIC, FEATURE_COLUMNS_CATEGORICAL
    leaked = {"isFlaggedFraud", "nameOrig", "nameDest", "newbalanceOrig", "newbalanceDest"}
    used = set(FEATURE_COLUMNS_NUMERIC) | set(FEATURE_COLUMNS_CATEGORICAL)
    assert leaked.isdisjoint(used), "leakage-prone raw columns must not be used as model features"
