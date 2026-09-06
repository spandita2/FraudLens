"""
Database connectivity tests.

These require a live PostgreSQL instance reachable via the DATABASE_URL /
POSTGRES_* env vars (see .env.example) and are skipped automatically if the
`sqlalchemy` package or a live database isn't available — they're meant to
be run in an environment with Postgres actually running (e.g. CI with a
postgres service container, or locally via docker-compose).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sqlalchemy = pytest.importorskip("sqlalchemy")


def _db_reachable() -> bool:
    try:
        from src.database.connection import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_reachable(), reason="No live PostgreSQL instance reachable.")


def test_connection_executes_simple_query():
    from src.database.connection import engine
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
    assert result == 1


def test_transactions_table_exists():
    from src.database.connection import engine
    from sqlalchemy import inspect
    inspector = inspect(engine)
    assert "transactions" in inspector.get_table_names()
