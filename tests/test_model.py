import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.feature_engineering import engineer_features

MODEL_PATH = ROOT / "models" / "fraudlens_pipeline.joblib"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not trained yet — run `python -m src.models.train_model` first.",
)


@pytest.fixture
def model_bundle():
    return joblib.load(MODEL_PATH)


def _raw_txn(**overrides):
    base = {
        "step": 10, "type": "TRANSFER", "amount": 181000.0,
        "nameOrig": "C1231006815", "oldbalanceOrg": 181000.0, "newbalanceOrig": 0.0,
        "nameDest": "C1666544295", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        "isFraud": 0, "isFlaggedFraud": 0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_model_bundle_has_expected_keys(model_bundle):
    assert "pipeline" in model_bundle
    assert "feature_columns" in model_bundle
    assert "model_name" in model_bundle


def test_predict_proba_in_valid_range(model_bundle):
    pipeline = model_bundle["pipeline"]
    features = engineer_features(_raw_txn())
    X = features[model_bundle["feature_columns"]]
    proba = pipeline.predict_proba(X)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_predict_returns_binary_class(model_bundle):
    pipeline = model_bundle["pipeline"]
    features = engineer_features(_raw_txn())
    X = features[model_bundle["feature_columns"]]
    pred = pipeline.predict(X)[0]
    assert pred in (0, 1)


def test_account_draining_pattern_scores_higher_than_small_payment(model_bundle):
    """Sanity check: a classic 'drain the account via TRANSFER' pattern
    should score meaningfully higher than an ordinary small PAYMENT."""
    pipeline = model_bundle["pipeline"]
    cols = model_bundle["feature_columns"]

    suspicious = engineer_features(_raw_txn(
        type="TRANSFER", amount=181000.0, oldbalanceOrg=181000.0,
    ))
    ordinary = engineer_features(_raw_txn(
        type="PAYMENT", amount=45.0, oldbalanceOrg=5000.0, nameDest="M998877",
    ))

    p_suspicious = pipeline.predict_proba(suspicious[cols])[0, 1]
    p_ordinary = pipeline.predict_proba(ordinary[cols])[0, 1]
    assert p_suspicious > p_ordinary
