"""
API tests using FastAPI's TestClient.

NOTE: these require `fastapi`, `httpx`, and the trained model artifact to be
present. They are skipped automatically if fastapi isn't installed, so the
rest of the test suite (preprocessing, model) can still run in stripped-down
environments.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

MODEL_PATH = ROOT / "models" / "fraudlens_pipeline.joblib"
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not trained yet — run `python -m src.models.train_model` first.",
)


@pytest.fixture
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "model_loaded" in body
    assert body["model_loaded"] is True


def test_predict_endpoint_valid_transaction(client):
    payload = {
        "step": 10, "type": "TRANSFER", "amount": 181000.0,
        "nameOrig": "C1231006815", "oldbalanceOrg": 181000.0,
        "nameDest": "C1666544295", "oldbalanceDest": 0.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["fraud_prediction"] in (0, 1)
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_predict_endpoint_rejects_invalid_type(client):
    payload = {
        "step": 10, "type": "NOT_A_REAL_TYPE", "amount": 100.0,
        "nameOrig": "C1", "oldbalanceOrg": 500.0, "nameDest": "C2",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422  # Pydantic validation error


def test_predict_endpoint_rejects_negative_amount(client):
    payload = {
        "step": 10, "type": "PAYMENT", "amount": -50.0,
        "nameOrig": "C1", "oldbalanceOrg": 500.0, "nameDest": "M1",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
