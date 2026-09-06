"""
preprocessing.py — cleaning + sklearn ColumnTransformer pipeline.

Cleaning steps (driven directly by eda.py findings):
  - Drop exact duplicate rows.
  - Missing oldbalanceDest -> median-imputed inside the sklearn Pipeline
    (fit on train only, applied to test — no leakage).

The ColumnTransformer (numeric scaling + categorical one-hot) is fit ONLY
on the training split; the same fitted transformer is applied to the test
split and saved inside the final joblib pipeline so inference-time
preprocessing exactly matches training-time preprocessing.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.features.feature_engineering import (
    FEATURE_COLUMNS_NUMERIC,
    FEATURE_COLUMNS_CATEGORICAL,
)


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"[clean_raw] Dropped {removed} exact duplicate rows.")
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, FEATURE_COLUMNS_NUMERIC),
        ("cat", categorical_pipeline, FEATURE_COLUMNS_CATEGORICAL),
    ])
    return preprocessor
