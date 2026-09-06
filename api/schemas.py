"""Pydantic request/response models for the FraudLens API."""
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator

VALID_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


class TransactionRequest(BaseModel):
    """Input required to score a single transaction, matching the raw
    PaySim-schema fields the model's feature pipeline needs. newbalance*
    fields are intentionally NOT accepted — see leakage note in
    src/data/eda.py: post-transaction balances aren't known at the moment
    a transaction must be scored."""

    step: int = Field(..., ge=0, description="Simulated hour index (0..743 in training range)")
    type: str = Field(..., description="One of CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER")
    amount: float = Field(..., gt=0, description="Transaction amount")
    nameOrig: str = Field(..., min_length=1)
    oldbalanceOrg: float = Field(..., ge=0)
    nameDest: str = Field(..., min_length=1)
    oldbalanceDest: Optional[float] = Field(default=0.0, ge=0)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in VALID_TYPES:
            raise ValueError(f"type must be one of {sorted(VALID_TYPES)}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "step": 10,
                "type": "TRANSFER",
                "amount": 181000.0,
                "nameOrig": "C1231006815",
                "oldbalanceOrg": 181000.0,
                "nameDest": "C1666544295",
                "oldbalanceDest": 0.0,
            }
        }


class PredictionResponse(BaseModel):
    fraud_prediction: int = Field(..., description="1 = predicted fraud, 0 = predicted legitimate")
    fraud_probability: float = Field(..., description="Model probability of fraud, 0-1")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH, derived from fraud_probability")
    model_used: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool


class TransactionOut(BaseModel):
    id: int
    step: int
    type: str
    amount: float
    name_orig: str
    name_dest: str
    is_fraud: int
    predicted_fraud: Optional[int] = None
    fraud_probability: Optional[float] = None

    class Config:
        from_attributes = True


class TransactionListResponse(BaseModel):
    count: int
    results: List[TransactionOut]
