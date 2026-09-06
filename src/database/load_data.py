"""
load_data.py — loads data/raw/transactions.csv into PostgreSQL.

Requires a running PostgreSQL instance and sql/schema.sql already applied
(or run create_all() below, which does the same thing from the ORM models).

Run: python -m src.database.load_data
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.database.connection import engine, SessionLocal
from src.database.models import Base, Transaction

RAW_PATH = ROOT / "data" / "raw" / "transactions.csv"
BATCH_SIZE = 5000


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Tables ensured (created if not already present).")


def load_csv_to_db(path: Path = RAW_PATH, batch_size: int = BATCH_SIZE):
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)

    session = SessionLocal()
    try:
        total = len(df)
        for start in range(0, total, batch_size):
            chunk = df.iloc[start:start + batch_size]
            rows = [
                Transaction(
                    step=int(r.step),
                    type=r.type,
                    amount=float(r.amount),
                    name_orig=r.nameOrig,
                    oldbalance_org=float(r.oldbalanceOrg),
                    newbalance_orig=float(r.newbalanceOrig),
                    name_dest=r.nameDest,
                    oldbalance_dest=None if pd.isna(r.oldbalanceDest) else float(r.oldbalanceDest),
                    newbalance_dest=float(r.newbalanceDest),
                    is_fraud=int(r.isFraud),
                    is_flagged_fraud=int(r.isFlaggedFraud),
                )
                for r in chunk.itertuples(index=False)
            ]
            session.bulk_save_objects(rows)
            session.commit()
            print(f"Loaded {min(start + batch_size, total):,}/{total:,} rows...")
        print("Load complete.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    create_tables()
    load_csv_to_db()
