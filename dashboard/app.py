"""
FraudLens Streamlit Dashboard.

Run: streamlit run dashboard/app.py

Data sources:
  - "Analytics" tab reads data/processed/transactions_features.csv directly
    (fast, no DB dependency, works right after training).
  - "Database Analytics" tab queries PostgreSQL through the FastAPI
    /transactions endpoint, so it only works once the API + DB are running.
  - "Risk Prediction" tab calls the FastAPI /predict endpoint live.
All KPIs and charts are computed from real loaded data — nothing here is
hand-typed or fabricated.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PROCESSED_PATH = ROOT / "data" / "processed" / "transactions_features.csv"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="FraudLens", page_icon="🛡️", layout="wide")

CUSTOM_CSS = """
<style>
.kpi-card {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
    padding: 1.1rem 1.3rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.kpi-label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em; }
.kpi-value { font-size: 1.7rem; font-weight: 700; color: #111827; }
.risk-high { color: #b91c1c; font-weight: 700; }
.risk-medium { color: #b45309; font-weight: 700; }
.risk-low { color: #15803d; font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PROCESSED_PATH)


@st.cache_data
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


def kpi_card(label: str, value: str):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def risk_badge(level: str) -> str:
    cls = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}.get(level, "")
    return f'<span class="{cls}">{level}</span>'


st.title("🛡️ FraudLens")
st.caption("Digital Payment Fraud Detection & Risk Analytics Platform")

df = load_processed_data()
metrics = load_metrics()

if df.empty:
    st.warning(
        "No processed data found. Run the pipeline first:\n\n"
        "`python -m src.models.train_model`"
    )
    st.stop()

tab_overview, tab_analytics, tab_predict, tab_db = st.tabs(
    ["Overview", "Analytics", "Risk Prediction", "Database Analytics"]
)

# ----------------------------------------------------------------- Overview
with tab_overview:
    total_txns = len(df)
    fraud_txns = int(df["isFraud"].sum())
    fraud_rate = fraud_txns / total_txns if total_txns else 0
    avg_amount = df["amount"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Transactions", f"{total_txns:,}")
    with c2: kpi_card("Fraudulent Transactions", f"{fraud_txns:,}")
    with c3: kpi_card("Fraud Rate", f"{fraud_rate:.3%}")
    with c4: kpi_card("Avg Transaction Amount", f"${avg_amount:,.2f}")

    st.markdown("###")
    if metrics:
        final_model = metrics.get("final_model", "n/a")
        model_metrics = metrics.get("models", {}).get(final_model, {})
        st.subheader(f"Deployed model: `{final_model}`")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        for col, key, label in zip(
            [mc1, mc2, mc3, mc4, mc5],
            ["precision", "recall", "f1_score", "roc_auc", "pr_auc"],
            ["Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"],
        ):
            with col:
                kpi_card(label, f"{model_metrics.get(key, 0):.4f}")
        st.caption(
            "Metrics computed on a held-out 20% test split. Dataset is a "
            "schema-faithful synthetic stand-in for the Kaggle PaySim "
            "dataset — see README for details."
        )

# ----------------------------------------------------------------- Analytics
with tab_analytics:
    left, right = st.columns(2)

    with left:
        st.subheader("Fraud vs Legitimate")
        dist = df["isFraud"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
        fig = px.pie(values=dist.values, names=dist.index, hole=0.45,
                     color=dist.index, color_discrete_map={"Legitimate": "#2563eb", "Fraud": "#dc2626"})
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Transactions by Type")
        type_counts = df["type"].value_counts()
        fig = px.bar(x=type_counts.index, y=type_counts.values,
                      labels={"x": "Type", "y": "Count"}, color=type_counts.index)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud Rate by Transaction Type")
    fraud_by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
    fig = px.bar(x=fraud_by_type.index, y=fraud_by_type.values,
                  labels={"x": "Type", "y": "Fraud Rate"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Transaction Volume Over Time (by simulated day)")
    trend = df.groupby("day").size()
    st.plotly_chart(px.line(x=trend.index, y=trend.values,
                             labels={"x": "Day", "y": "Transaction Count"}), use_container_width=True)

    st.subheader("Fraud by Hour of Day")
    fraud_hourly = df[df["isFraud"] == 1].groupby("hour_of_day").size()
    st.plotly_chart(px.bar(x=fraud_hourly.index, y=fraud_hourly.values,
                            labels={"x": "Hour of Day", "y": "Fraud Count"}), use_container_width=True)

    st.subheader("Transaction Amount Distribution (log scale)")
    fig = px.histogram(df, x="amount_log", color=df["isFraud"].map({0: "Legitimate", 1: "Fraud"}),
                        barmode="overlay", nbins=60)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------- Risk Prediction
with tab_predict:
    st.subheader("Score a Transaction")
    st.caption(f"Sends a request to the FastAPI backend at `{API_BASE_URL}/predict`.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            step = st.number_input("Step (hour index)", min_value=0, max_value=743, value=10)
            txn_type = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
        with c2:
            amount = st.number_input("Amount", min_value=0.01, value=5000.0, step=100.0)
            old_balance_org = st.number_input("Origin Old Balance", min_value=0.0, value=10000.0, step=100.0)
        with c3:
            old_balance_dest = st.number_input("Destination Old Balance", min_value=0.0, value=0.0, step=100.0)
        name_orig = st.text_input("Origin Customer ID", value="C1231006815")
        name_dest = st.text_input("Destination ID", value="C1666544295")
        submitted = st.form_submit_button("Score Transaction")

    if submitted:
        payload = {
            "step": step, "type": txn_type, "amount": amount,
            "nameOrig": name_orig, "oldbalanceOrg": old_balance_org,
            "nameDest": name_dest, "oldbalanceDest": old_balance_dest,
        }
        try:
            resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()
            c1, c2, c3 = st.columns(3)
            with c1:
                kpi_card("Prediction", "FRAUD" if result["fraud_prediction"] == 1 else "LEGITIMATE")
            with c2:
                kpi_card("Fraud Probability", f"{result['fraud_probability']:.2%}")
            with c3:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">Risk Level</div>'
                    f'<div class="kpi-value">{risk_badge(result["risk_level"])}</div></div>',
                    unsafe_allow_html=True,
                )
        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not reach the FastAPI backend at {API_BASE_URL}. "
                f"Start it with: `uvicorn api.main:app --reload --port 8000`"
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"API returned an error: {e.response.text}")

# --------------------------------------------------------------- Database Analytics
with tab_db:
    st.subheader("Live PostgreSQL-backed Data")
    st.caption(f"Queries the FastAPI backend, which reads from PostgreSQL, at `{API_BASE_URL}/transactions`.")

    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        limit = st.slider("Rows to fetch", 10, 500, 100)
    with colf2:
        fraud_filter = st.selectbox("Fraud filter", ["All", "Fraud only", "Legitimate only"])
    with colf3:
        type_filter = st.selectbox("Type filter", ["All", "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

    params = {"limit": limit}
    if fraud_filter == "Fraud only":
        params["is_fraud"] = 1
    elif fraud_filter == "Legitimate only":
        params["is_fraud"] = 0
    if type_filter != "All":
        params["type"] = type_filter

    if st.button("Fetch from database"):
        try:
            resp = requests.get(f"{API_BASE_URL}/transactions", params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            st.success(f"Fetched {data['count']} rows from PostgreSQL.")
            st.dataframe(pd.DataFrame(data["results"]), use_container_width=True)
        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not reach the FastAPI backend / database at {API_BASE_URL}. "
                f"Ensure PostgreSQL is running, data is loaded "
                f"(`python -m src.database.load_data`), and the API is up."
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"API returned an error: {e.response.text}")
