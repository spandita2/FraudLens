"""SQLAlchemy ORM model mirroring sql/schema.sql exactly."""
from sqlalchemy import Column, BigInteger, Integer, SmallInteger, String, Numeric, DateTime, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    step = Column(Integer, nullable=False)
    type = Column(String(20), nullable=False)
    amount = Column(Numeric(18, 2), nullable=False)
    name_orig = Column(String(30), nullable=False)
    oldbalance_org = Column(Numeric(18, 2))
    newbalance_orig = Column(Numeric(18, 2))
    name_dest = Column(String(30), nullable=False)
    oldbalance_dest = Column(Numeric(18, 2))
    newbalance_dest = Column(Numeric(18, 2))
    is_fraud = Column(SmallInteger, nullable=False, default=0)
    is_flagged_fraud = Column(SmallInteger, nullable=False, default=0)
    predicted_fraud = Column(SmallInteger, nullable=True)
    fraud_probability = Column(Numeric(6, 5), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "step": self.step,
            "type": self.type,
            "amount": float(self.amount) if self.amount is not None else None,
            "name_orig": self.name_orig,
            "oldbalance_org": float(self.oldbalance_org) if self.oldbalance_org is not None else None,
            "newbalance_orig": float(self.newbalance_orig) if self.newbalance_orig is not None else None,
            "name_dest": self.name_dest,
            "oldbalance_dest": float(self.oldbalance_dest) if self.oldbalance_dest is not None else None,
            "newbalance_dest": float(self.newbalance_dest) if self.newbalance_dest is not None else None,
            "is_fraud": self.is_fraud,
            "is_flagged_fraud": self.is_flagged_fraud,
            "predicted_fraud": self.predicted_fraud,
            "fraud_probability": float(self.fraud_probability) if self.fraud_probability is not None else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
