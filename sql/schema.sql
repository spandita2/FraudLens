-- FraudLens PostgreSQL schema
-- Run: psql -U <user> -d <dbname> -f sql/schema.sql

CREATE TABLE IF NOT EXISTS transactions (
    id                  BIGSERIAL PRIMARY KEY,
    step                INTEGER NOT NULL,
    type                VARCHAR(20) NOT NULL,
    amount              NUMERIC(18, 2) NOT NULL,
    name_orig           VARCHAR(30) NOT NULL,
    oldbalance_org      NUMERIC(18, 2),
    newbalance_orig     NUMERIC(18, 2),
    name_dest           VARCHAR(30) NOT NULL,
    oldbalance_dest     NUMERIC(18, 2),
    newbalance_dest     NUMERIC(18, 2),
    is_fraud            SMALLINT NOT NULL DEFAULT 0,
    is_flagged_fraud    SMALLINT NOT NULL DEFAULT 0,
    predicted_fraud     SMALLINT,             -- filled in by the API for scored transactions
    fraud_probability   NUMERIC(6, 5),        -- filled in by the API for scored transactions
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes chosen to match actual dashboard/API query patterns:
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions (type);
CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud ON transactions (is_fraud);
CREATE INDEX IF NOT EXISTS idx_transactions_step ON transactions (step);
CREATE INDEX IF NOT EXISTS idx_transactions_name_orig ON transactions (name_orig);
CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions (created_at);

COMMENT ON TABLE transactions IS 'Digital payment transactions scored by FraudLens.';
COMMENT ON COLUMN transactions.predicted_fraud IS 'Model prediction (0/1), NULL if not yet scored via API.';
COMMENT ON COLUMN transactions.fraud_probability IS 'Model-estimated probability of fraud, 0 to 1.';
