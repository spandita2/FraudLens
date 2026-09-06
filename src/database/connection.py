"""
Database engine/session factory. Reads config from environment variables
(see .env.example) — never hard-codes credentials.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can also be set directly

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}".format(
        user=os.getenv("POSTGRES_USER", "fraudlens_user"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        db=os.getenv("POSTGRES_DB", "fraudlens"),
    ),
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    """FastAPI dependency: yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
