"""
FraudLens FastAPI backend.

Run: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.schemas import (
    TransactionRequest, PredictionResponse, HealthResponse,
    TransactionOut, TransactionListResponse,
)
from src.database.connection import get_db, engine
from src.database.models import Transaction
from src.features.feature_engineering import add_row_level_features

MODEL_PATH = ROOT / "models" / "fraudlens_pipeline.joblib"

_model_bundle = {"pipeline": None, "history_encoder": None, "feature_columns": None, "model_name": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup (not per-request).
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        _model_bundle.update(bundle)
        print(f"Loaded model pipeline: {_model_bundle['model_name']}")
    else:
        print(f"WARNING: model file not found at {MODEL_PATH}. "
              f"Run `python -m src.models.train_model` first. /predict will fail until then.")
    yield
    _model_bundle["pipeline"] = None


app = FastAPI(
    title="FraudLens API",
    description="Digital payment fraud detection & risk scoring API.",
    version="1.0.0",
    lifespan=lifespan,
)


def _risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "HIGH"
    if prob >= 0.25:
        return "MEDIUM"
    return "LOW"


@app.get("/health", response_model=HealthResponse)
def health():
    db_ok = True
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    return HealthResponse(
        status="ok" if _model_bundle["pipeline"] is not None else "degraded",
        model_loaded=_model_bundle["pipeline"] is not None,
        database_connected=db_ok,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: TransactionRequest):
    pipeline = _model_bundle["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run `python -m src.models.train_model` "
                   "to produce models/fraudlens_pipeline.joblib, then restart the API.",
        )

    # Build a single-row raw frame with the exact PaySim columns the feature
    # engineering step expects. newbalance* fields are set to NaN-safe
    # placeholders since they are never used as model inputs (see schemas.py).
    raw = pd.DataFrame([{
        "step": txn.step,
        "type": txn.type,
        "amount": txn.amount,
        "nameOrig": txn.nameOrig,
        "oldbalanceOrg": txn.oldbalanceOrg,
        "newbalanceOrig": max(txn.oldbalanceOrg - txn.amount, 0.0),  # placeholder, unused by model
        "nameDest": txn.nameDest,
        "oldbalanceDest": txn.oldbalanceDest if txn.oldbalanceDest is not None else 0.0,
        "newbalanceDest": 0.0,  # placeholder, unused by model
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }])

    try:
        features = add_row_level_features(raw)
        history_encoder = _model_bundle["history_encoder"]
        # Look up (don't update) this customer's history: an unconfirmed,
        # not-yet-scored transaction shouldn't count as history yet — the
        # same discipline `transform()` uses for the chronological test set.
        txn_count, avg_amount = history_encoder.transform_single(txn.nameOrig)
        features["orig_txn_count"] = txn_count
        features["orig_avg_amount"] = avg_amount

        feature_columns = _model_bundle["feature_columns"]
        X = features[feature_columns]
        proba = float(pipeline.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to score transaction: {e}")

    return PredictionResponse(
        fraud_prediction=pred,
        fraud_probability=round(proba, 5),
        risk_level=_risk_level(proba),
        model_used=_model_bundle["model_name"] or "unknown",
    )


@app.get("/transactions", response_model=TransactionListResponse)
def get_transactions(
    limit: int = 50,
    is_fraud: int | None = None,
    type: str | None = None,
    db: Session = Depends(get_db),
):
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    query = db.query(Transaction)
    if is_fraud is not None:
        query = query.filter(Transaction.is_fraud == is_fraud)
    if type is not None:
        query = query.filter(Transaction.type == type.upper())
    rows = query.order_by(Transaction.id.desc()).limit(limit).all()
    return TransactionListResponse(
        count=len(rows),
        results=[TransactionOut.model_validate(r) for r in rows],
    )